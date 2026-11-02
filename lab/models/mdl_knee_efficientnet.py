import hashlib
from pathlib import Path

import timm
import torch
from torch import nn

from mdl_knee_2p5d import optional_depth_mixer


class EfficientNet25D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        kwargs = {}
        if cfg.encoder_3d_pretrained:
            kwargs["pretrained_cfg_overlay"] = {"file": cfg.encoder_weights}
        self.backbone = timm.create_model(
            cfg.encoder_2d,
            pretrained=cfg.encoder_3d_pretrained,
            in_chans=len(cfg.slice_offsets),
            num_classes=0,
            **kwargs,
        )
        self.out_channels = int(self.backbone.num_features)
        self.channels_last = cfg.channels_last
        self.native_triplets = getattr(cfg, "native_triplets", False)
        self.center_stride = getattr(cfg, "slice_center_stride", 1)
        self.imagenet_normalization = getattr(
            cfg, "encoder_imagenet_normalization", True)
        self.register_buffer(
            "slice_offsets", torch.tensor(cfg.slice_offsets), persistent=False)
        pretrained = self.backbone.pretrained_cfg
        channels = torch.arange(len(cfg.slice_offsets)) % 3
        self.register_buffer(
            "mean", torch.tensor(pretrained["mean"])[channels].view(1, -1, 1, 1),
            persistent=False)
        self.register_buffer(
            "std", torch.tensor(pretrained["std"])[channels].view(1, -1, 1, 1),
            persistent=False)
        self.depth_mixer = optional_depth_mixer(cfg, self.out_channels)
        self.accepts_slot_ids = True
        if getattr(cfg, "gradient_checkpointing", True):
            if not hasattr(self.backbone, "set_grad_checkpointing"):
                raise RuntimeError(
                    "EfficientNet backbone lacks gradient checkpointing support")
            self.backbone.set_grad_checkpointing(True)

    def forward_features(self, x, slot_ids=None):
        n, _, depth, height, width = x.shape
        centers = torch.arange(
            0, depth, self.center_stride, device=x.device)
        if self.native_triplets:
            images = x[:, :, centers].permute(0, 2, 1, 3, 4)
        else:
            indices = centers.unsqueeze(1) + self.slice_offsets
            images = x[:, 0][:, indices.clamp(0, depth - 1)]
        images = images.reshape(
            n * len(centers), len(self.slice_offsets), height, width)
        if self.imagenet_normalization:
            images = (images - self.mean) / self.std
        if self.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        features = self.backbone.forward_features(images)
        features = features.reshape(
            n, len(centers), self.out_channels,
            *features.shape[-2:]).permute(0, 2, 1, 3, 4)
        if self.depth_mixer is not None:
            features = self.depth_mixer(features, slot_ids)
        return features


def build_encoder(cfg):
    if cfg.encoder_3d_pretrained:
        path = Path(cfg.encoder_weights)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != cfg.encoder_weights_sha256:
            raise ValueError(
                f"EfficientNet pretrained checkpoint hash mismatch: {path}")
    encoder = EfficientNet25D(cfg)
    return encoder, encoder.out_channels
