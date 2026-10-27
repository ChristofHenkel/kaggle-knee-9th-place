import math
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
SLOTS = ['AXI_FS', 'AXI_NFS', 'COR_FS', 'COR_NFS_FL', 'COR_NFS_ST', 'SAG_FS', 'SAG_NFS_FL', 'SAG_NFS_ST']
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
KINETICS_MEAN = 0.40109
KINETICS_STD = 0.22216

def _named_child(root, name):
    obj = root
    for part in name.split('.'):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj

def _replace_module(root, name, new):
    parts = name.split('.')
    parent = _named_child(root, '.'.join(parts[:-1])) if len(parts) > 1 else root
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)

def _sum_stem_to_1ch(conv):
    new = nn.Conv3d(1, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new

def _bn3d_to_groupnorm(model, groups=32):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.BatchNorm3d):
            g = groups
            while g > 1 and mod.num_features % g != 0:
                g //= 2
            gn = nn.GroupNorm(g, mod.num_features, eps=mod.eps, affine=True)
            with torch.no_grad():
                gn.weight.copy_(mod.weight)
                gn.bias.copy_(mod.bias)
            _replace_module(model, name, gn)
    return model

def build_encoder_3d(cfg):
    if cfg.encoder_3d == 'slow_r50':
        from mdl_knee_slow import build_encoder
    elif cfg.encoder_3d == 'convnext_2p5d':
        from mdl_knee_convnext import build_encoder
    else:
        raise ValueError(cfg.encoder_3d)
    return build_encoder(cfg)

def encoder_features(encoder, x):
    if hasattr(encoder, 'forward_features'):
        return encoder.forward_features(x)
    multiscale = getattr(encoder, 'multiscale', False)
    h = encoder.stem(x)
    h = encoder.layer1(h)
    h = encoder.layer2(h)
    if multiscale:
        f28 = h
    h = encoder.layer3(h)
    if multiscale:
        f14 = h
    h = encoder.layer4(h)
    return (f28, f14, h) if multiscale else h

class GeM3D(nn.Module):

    def __init__(self, p, eps):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, feat, smask):
        p = self.p.clamp_min(self.eps)
        weights = smask.float()
        weights = torch.where(weights.sum(-1, keepdim=True) > 0, weights, torch.ones_like(weights))
        log_values = feat.float().clamp_min(self.eps).log() * p
        log_values = log_values.masked_fill(weights[:, None, :, None, None] <= 0, -torch.inf)
        count = weights.sum(-1, keepdim=True) * feat.shape[-2] * feat.shape[-1]
        log_mean = torch.logsumexp(log_values, dim=(2, 3, 4)) - count.log()
        return (log_mean / p).exp()

def masked_pool(feat, smask, q=None, gate=None, spatial_q=None, spatial_gate=None, stats_out=None):
    if isinstance(feat, tuple):
        channels = [level.shape[1] for level in feat]
        queries = q.split(channels, dim=1) if q is not None else [None] * len(feat)
        spatial_queries = spatial_q.split(channels, dim=1) if spatial_q is not None else [None] * len(feat)
        return torch.cat([masked_pool(level, smask, level_q, gate, level_spatial_q, spatial_gate, stats_out if index == len(feat) - 1 else None) for index, (level, level_q, level_spatial_q) in enumerate(zip(feat, queries, spatial_queries))], dim=-1)
    f = feat.mean(dim=(3, 4))
    w = smask.to(f.dtype)
    if w.shape[-1] != f.shape[-1]:
        w = F.adaptive_avg_pool1d(w.unsqueeze(1), f.shape[-1]).squeeze(1)
    tot = w.sum(-1, keepdim=True)
    w = torch.where(tot > 0, w, torch.ones_like(w))
    base = (f * w.unsqueeze(1)).sum(-1) / w.sum(-1, keepdim=True)
    if q is None:
        return base
    att = torch.einsum('kc,nct->nkt', q, f) / math.sqrt(f.shape[1])
    att = att.masked_fill(w.unsqueeze(1) <= 0, -10000.0).softmax(-1)
    attended = torch.einsum('nkt,nct->nkc', att, f)
    g = gate.to(f.dtype).view(1, -1, 1)
    out = base.unsqueeze(1) + g * (attended - base.unsqueeze(1))
    if spatial_q is None:
        return out
    sp_scores = torch.einsum('kc,ncthw->nkthw', spatial_q, feat) / math.sqrt(feat.shape[1])
    n, kk, t, h, wd = sp_scores.shape
    sp_att_flat = sp_scores.reshape(n, kk, t, h * wd).softmax(-1)
    if stats_out is not None:
        with torch.no_grad():
            real = (w > 0).to(sp_att_flat.dtype)
            maxw = sp_att_flat.amax(-1)
            ent = -(sp_att_flat.clamp_min(1e-12) * sp_att_flat.clamp_min(1e-12).log()).sum(-1) / math.log(h * wd)
            r = real.unsqueeze(1)
            stats_out['spatial_maxw_sum'] = (maxw * r).sum(dim=(0, 2))
            stats_out['spatial_entropy_sum'] = (ent * r).sum(dim=(0, 2))
            stats_out['spatial_n'] = real.sum()
    sp_att = sp_att_flat.reshape(n, kk, t, h, wd)
    f_spatial = torch.einsum('nkthw,ncthw->nkct', sp_att, feat)
    spatial_attended = torch.einsum('nkt,nkct->nkc', att, f_spatial)
    sg = spatial_gate.to(f.dtype).view(1, -1, 1)
    return out + sg * (spatial_attended - out)

def encode_packed(cfg, encoder, x, live, smask, q=None, gate=None, spatial_q=None, spatial_gate=None, stats_out=None, pooler=None):
    smask = smask[:, ::getattr(encoder, 'center_stride', 1)]
    if pooler is None:
        pooler = partial(masked_pool, q=q, gate=gate, spatial_q=spatial_q, spatial_gate=spatial_gate, stats_out=stats_out)
    if not cfg.pack_live_slots:
        return pooler(encoder_features(encoder, x), smask)
    if not bool(live.any()):
        live = live.clone()
        live[0] = True
    out = pooler(encoder_features(encoder, x[live]), smask[live])
    full = out.new_zeros((x.shape[0], *out.shape[1:]))
    full[live] = out
    return full

class SlotMIL(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder, c = build_encoder_3d(cfg)
        e = self.cfg.embed_dim
        self.num_slots = len(getattr(cfg, 'slot_names', SLOTS))
        self.proj = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, e), nn.GELU(), nn.Dropout(self.cfg.dropout)) if getattr(cfg, 'feature_projection', True) else nn.Identity()
        self.slot_embed = nn.Parameter(torch.zeros(self.num_slots, e)) if getattr(cfg, 'slot_embedding', True) else None
        self.tgt_query = nn.Parameter(torch.zeros(len(TARGETS), e))
        self.head_w = nn.Parameter(torch.zeros(len(TARGETS), e))
        self.head_b = nn.Parameter(torch.zeros(len(TARGETS)))
        for p in (self.slot_embed, self.tgt_query, self.head_w):
            if p is not None:
                nn.init.trunc_normal_(p, std=0.02)
        self.att_proj = nn.Linear(e, e)
        self.aux_head = nn.Linear(c, len(TARGETS))
        if getattr(cfg, 'volume_pooling', 'attention') == 'gem':
            self.volume_pooler = GeM3D(cfg.gem_p_init, cfg.gem_eps)
        else:
            self.slice_q = nn.Parameter(torch.zeros(len(TARGETS), c))
            nn.init.trunc_normal_(self.slice_q, std=0.02)
            self.slice_gate = nn.Parameter(torch.full((len(TARGETS),), float(self.cfg.slice_attn_gate_init)))
            self.spatial_q = nn.Parameter(torch.zeros(len(TARGETS), c))
            nn.init.trunc_normal_(self.spatial_q, std=0.02)
            self.spatial_gate = nn.Parameter(torch.full((len(TARGETS),), float(self.cfg.spatial_attn_gate_init)))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.cfg.norm3d == 'freeze_bn':
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
        return self

    def forward(self, x, mask, smask, stats_out=None):
        x = torch.nan_to_num(x, 0.0, 0.0, 0.0)
        b = mask.shape[0]
        s = self.slot_embed.shape[0] if self.slot_embed is not None else self.num_slots
        if x.shape[0] != b * s:
            raise ValueError(f'x batch {x.shape[0]} is not b*s = {b}*{s}; v016 feeds one VOLUME per series, not one image per slice')
        if smask.shape[:2] != (b, s):
            raise ValueError(f'slice mask {tuple(smask.shape)} is not (B,S,D) = ({b},{s},{x.shape[2]})')
        _q = self.slice_q if self.cfg.slice_attn else None
        _g = self.slice_gate if self.cfg.slice_attn else None
        _sq = self.spatial_q if self.cfg.spatial_attn else None
        _sg = self.spatial_gate if self.cfg.spatial_attn else None
        feats = encode_packed(self.cfg, self.encoder, x, mask.reshape(-1) > 0, smask.reshape(b * s, -1), _q, _g, _sq, _sg, stats_out, pooler=getattr(self, 'volume_pooler', None))
        feats = feats.view(b, s, *feats.shape[1:])
        if self.cfg.slice_attn:
            aux = torch.einsum('bskc,kc->bsk', feats, self.aux_head.weight) + self.aux_head.bias
        else:
            aux = self.aux_head(feats)
        aux = aux.masked_fill(mask.unsqueeze(-1) == 0, -10000.0).amax(1)
        aux = torch.nan_to_num(aux, 0.0, 0.0, 0.0)
        slot_vec = self.proj(feats)
        if self.slot_embed is not None:
            _se = self.slot_embed.view(1, s, 1, -1) if self.cfg.slice_attn else self.slot_embed.unsqueeze(0)
            slot_vec = slot_vec + _se
        k = self.att_proj(slot_vec)
        if self.cfg.slice_attn:
            att = torch.einsum('te,bste->bts', self.tgt_query, k) / math.sqrt(k.shape[-1])
            att = att.masked_fill(mask.unsqueeze(1) == 0, -10000.0).softmax(-1)
            pooled = torch.einsum('bts,bste->bte', att, slot_vec)
        else:
            att = torch.einsum('te,bse->bts', self.tgt_query, k) / math.sqrt(k.shape[-1])
            att = att.masked_fill(mask.unsqueeze(1) == 0, -10000.0).softmax(-1)
            pooled = torch.einsum('bts,bse->bte', att, slot_vec)
        logits = (pooled * self.head_w.unsqueeze(0)).sum(-1) + self.head_b
        logits = torch.nan_to_num(logits, 0.0, 0.0, 0.0)
        return (logits, aux, att)

def to_float(cfg, imgs, smask=None):
    b, s, d, ch, h, w = imgs.shape
    expected_channels = getattr(cfg, 'input_channels', 1)
    if ch != expected_channels:
        raise ValueError(f'Expected {expected_channels} input channels, got {ch}')
    x = imgs.reshape(b * s, d, ch, h, w).float().div_(255.0)
    if getattr(cfg, 'input_normalization', 'kinetics') == 'foreground_zscore':
        weights = (x > 0) & (smask.reshape(b * s, d, 1, 1, 1) > 0)
        count = weights.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1)
        mean = (x * weights).sum(dim=(1, 2, 3, 4), keepdim=True) / count
        variance = ((x - mean).square() * weights).sum(dim=(1, 2, 3, 4), keepdim=True) / count
        x = ((x - mean) / variance.sqrt().clamp_min(1e-06)).masked_fill(~weights, 0)
    elif getattr(cfg, 'input_normalization', 'kinetics') == 'series_zscore':
        weights = smask.reshape(b * s, d, 1, 1, 1).to(x.dtype)
        count = (weights.sum(dim=1, keepdim=True) * h * w).clamp_min(1)
        mean = (x * weights).sum(dim=(1, 2, 3, 4), keepdim=True) / count
        variance = ((x - mean).square() * weights).sum(dim=(1, 2, 3, 4), keepdim=True) / count
        x = (x - mean) / variance.sqrt().clamp_min(1e-06)
    elif getattr(cfg, 'input_normalization', 'kinetics') == 'volume_zscore':
        variance, mean = torch.var_mean(x, dim=(1, 2, 3, 4), correction=0, keepdim=True)
        x = (x - mean) / variance.sqrt().clamp_min(1e-06)
    elif getattr(cfg, 'input_normalization', 'kinetics') != 'unit':
        x = (x - KINETICS_MEAN) / KINETICS_STD
    x = x.permute(0, 2, 1, 3, 4)
    if cfg.channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    else:
        x = x.contiguous()
    return x

def loss_fn(cfg, logits, aux, y, conf, loss_weight=None):
    w = conf if cfg.use_conf_weight else None
    if loss_weight is not None:
        w = loss_weight if w is None else w * loss_weight
    normalize = getattr(cfg, 'normalize_loss_weight', False)
    reduction = 'sum' if normalize else 'mean'
    loss = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction=reduction)
    if cfg.aux_weight > 0:
        loss = loss + cfg.aux_weight * F.binary_cross_entropy_with_logits(aux, y, weight=w, reduction=reduction)
    if normalize:
        denominator = w.sum()
        if cfg.distributed:
            torch.distributed.all_reduce(denominator)
            denominator = denominator / cfg.world_size
        loss = loss / denominator
    return loss

class Net(SlotMIL):

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.channels_last:
            memory_format = torch.channels_last if cfg.encoder_3d in ('convnext_2p5d', 'efficientnet_2p5d') else torch.channels_last_3d
            self.to(memory_format=memory_format)

    def optimizer_parameters(self):
        enc, head = ([], [])
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (enc if name.startswith('encoder.') else head).append(parameter)
        return [{'params': enc, 'lr': self.cfg.lr_backbone}, {'params': head, 'lr': self.cfg.lr_head}]

    def forward(self, batch):
        x = to_float(self.cfg, batch['images'], batch['smask'])
        logits, aux, attention = super().forward(x, batch['mask'], batch['smask'])
        if self.training:
            return {'loss': loss_fn(self.cfg, logits, aux, batch['target'], batch['confidence'], batch.get('loss_weight'))}
        return {'predictions': logits.float().sigmoid(), 'aux_predictions': aux.float().sigmoid()}
