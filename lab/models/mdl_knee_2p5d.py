import torch
from torch import nn
from torch.nn import functional as F


class DepthMixer(nn.Module):
    def __init__(self, cfg, channels):
        super().__init__()
        bottleneck = int(cfg.hybrid_bottleneck)
        groups = min(32, bottleneck)
        while groups > 1 and bottleneck % groups != 0:
            groups //= 2
        kernel = int(cfg.hybrid_depth_kernel)
        self.use_type_embedding = bool(cfg.type_embedding)
        self.depth_in = nn.Conv3d(channels, bottleneck, 1, bias=False)
        self.depth_norm = nn.GroupNorm(groups, bottleneck)
        self.depth_conv = nn.Conv3d(
            bottleneck, bottleneck, kernel_size=(kernel, 1, 1),
            padding=(kernel // 2, 0, 0), groups=bottleneck, bias=False)
        self.depth_out = nn.Conv3d(bottleneck, channels, 1, bias=False)
        nn.init.zeros_(self.depth_out.weight)
        self.hybrid_gate = nn.Parameter(torch.tensor(float(cfg.hybrid_gate_init)))
        self.type_embed = nn.Parameter(
            torch.zeros(len(cfg.slot_names), bottleneck))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, feature_map, slot_ids):
        if slot_ids is None:
            raise ValueError('DepthMixer requires global slot ids')
        z = self.depth_in(feature_map)
        if self.use_type_embedding:
            z = z + self.type_embed[slot_ids].view(z.shape[0], -1, 1, 1, 1)
        z = F.silu(self.depth_norm(z))
        z = F.silu(self.depth_conv(z))
        return feature_map + self.hybrid_gate.to(feature_map.dtype) * self.depth_out(z)


def optional_depth_mixer(cfg, channels):
    if int(getattr(cfg, 'hybrid_bottleneck', 0)) <= 0:
        return None
    return DepthMixer(cfg, channels)
