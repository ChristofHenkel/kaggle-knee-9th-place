import hashlib
from pathlib import Path

import timm
from timm.models.convnext import ConvNeXtStage
import torch
from torch import nn


class ConvNeXt25D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.encoder_2d,
            pretrained=cfg.encoder_3d_pretrained,
            in_chans=len(cfg.slice_offsets),
            features_only=True,
            out_indices=getattr(cfg, 'encoder_out_indices', (3,)),
            pretrained_cfg_overlay={'file': cfg.encoder_weights},
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]
        for stage in self.backbone.modules():
            if isinstance(stage, ConvNeXtStage):
                stage.grad_checkpointing = cfg.gradient_checkpointing
        self.channels_last = cfg.channels_last
        self.native_triplets = getattr(cfg, 'native_triplets', False)
        self.center_stride = getattr(cfg, 'slice_center_stride', 1)
        self.imagenet_normalization = getattr(cfg, 'encoder_imagenet_normalization', True)
        self.register_buffer('slice_offsets', torch.tensor(cfg.slice_offsets), persistent=False)
        pretrained = self.backbone.pretrained_cfg
        channels = torch.arange(len(cfg.slice_offsets)) % 3
        self.register_buffer('mean', torch.tensor(pretrained['mean'])[channels].view(1, -1, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(pretrained['std'])[channels].view(1, -1, 1, 1), persistent=False)

    def forward_features(self, x):
        n, _, depth, height, width = x.shape
        centers = torch.arange(0, depth, self.center_stride, device=x.device)
        if self.native_triplets:
            images = x[:, :, centers].permute(0, 2, 1, 3, 4)
        else:
            indices = centers.unsqueeze(1) + self.slice_offsets
            images = x[:, 0][:, indices.clamp(0, depth - 1)]
        images = images.reshape(n * len(centers), len(self.slice_offsets), height, width)
        if self.imagenet_normalization:
            images = (images - self.mean) / self.std
        if self.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        features = self.backbone(images)[0]
        return features.reshape(n, len(centers), self.out_channels, *features.shape[-2:]).permute(0, 2, 1, 3, 4)


def build_encoder(cfg):
    if cfg.encoder_3d_pretrained:
        path = Path(cfg.encoder_weights)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != cfg.encoder_weights_sha256:
            raise ValueError(f'Pretrained checkpoint hash mismatch: {path}')
    encoder = ConvNeXt25D(cfg)
    return encoder, encoder.out_channels
