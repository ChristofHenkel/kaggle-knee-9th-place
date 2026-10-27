import hashlib
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

from pytorchvideo.models.hub import slow_r50
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mdl_knee import _sum_stem_to_1ch


@contextmanager
def restore_batchnorm_buffers(module):
    buffers = [(buffer, buffer.clone()) for layer in module.modules()
               if isinstance(layer, nn.modules.batchnorm._BatchNorm)
               for buffer in (layer.running_mean, layer.running_var, layer.num_batches_tracked)]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)


def checkpoint_context(module):
    return nullcontext(), restore_batchnorm_buffers(module)


class CheckpointedEncoder(nn.Module):
    def forward_features(self, x):
        for stage in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            if self.training and torch.is_grad_enabled():
                x = checkpoint(stage, x, use_reentrant=False,
                               context_fn=partial(checkpoint_context, stage))
            else:
                x = stage(x)
        return x


def build_encoder(cfg):
    network = slow_r50(pretrained=False)
    if cfg.encoder_3d_pretrained:
        path = Path(cfg.encoder_weights)
        if hashlib.sha256(path.read_bytes()).hexdigest() != cfg.encoder_weights_sha256:
            raise ValueError("Pretrained checkpoint hash mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint['model_state']
        network.load_state_dict(state, strict=True)
    dimension = network.blocks[-1].proj.in_features
    network.blocks[0].conv = _sum_stem_to_1ch(network.blocks[0].conv)
    encoder = CheckpointedEncoder() if getattr(cfg, 'slow_stage_checkpointing', False) else nn.Module()
    encoder.stem, encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4 = list(network.blocks)[:-1]
    return encoder, dimension
