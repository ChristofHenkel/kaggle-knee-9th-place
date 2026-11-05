"""0363 image-space router with one mixed-shape, shared-parameter MaxViT.

Only channels-last timm MaxViT with MLP relative positions is supported.
Shape dispatch is local to each forward, including checkpoint recomputation.
"""

from torch.nn import functional as F
from timm.layers.pos_embed_rel import (
    RelPosMlp,
    gen_relative_log_coords,
    gen_relative_position_index,
)
from timm.models.maxxvit import (
    AttentionCl,
    PartitionAttentionCl,
    grid_partition,
    grid_reverse,
    window_partition,
    window_reverse,
)

from skp.models.classification.rsna_knee_maxvit_compartment_router import Net as Base


class MixedShapeAttention(AttentionCl):
    def forward(self, x, shared_rel_pos=None):
        size = tuple(x.shape[1:3])
        rel = self.rel_pos
        if size == tuple(rel.window_size):
            return super().forward(x, shared_rel_pos)
        if size != self.rect_partition or shared_rel_pos is not None:
            raise ValueError(f"Unsupported attention partition: {size}")
        # The same learned MLP evaluates a second, non-learned coordinate grid.
        bias = rel.mlp(self.rect_coords)
        area = size[0] * size[1]
        bias = bias.reshape(-1, rel.num_heads)[self.rect_index]
        bias = rel.bias_act(bias.reshape(area, area, rel.num_heads).permute(2, 0, 1))
        if rel.bias_gain is not None:
            bias = bias * rel.bias_gain
        bias = bias.unsqueeze(0).contiguous()
        batch = x.shape[0]
        if self.head_first:
            q, k, v = (
                self.qkv(x)
                .view(batch, -1, self.num_heads, self.dim_head * 3)
                .transpose(1, 2)
                .chunk(3, dim=3)
            )
        else:
            q, k, v = (
                self.qkv(x)
                .reshape(batch, -1, 3, self.num_heads, self.dim_head)
                .transpose(1, 3)
                .unbind(2)
            )
        if self.fused_attn:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attention = (q * self.scale) @ k.transpose(-2, -1) + bias
            output = self.attn_drop(attention.softmax(dim=-1)) @ v
        output = output.transpose(1, 2).reshape(x.shape[:-1] + (-1,))
        return self.proj_drop(self.proj(output))


class MixedShapePartition(PartitionAttentionCl):
    def _partition_attn(self, x):
        height, width = x.shape[1:3]
        if height == width:
            return super()._partition_attn(x)
        ph, pw = self.rect_partition
        if height * pw != width * ph:
            raise ValueError(f"Unsupported feature aspect ratio: {(height, width)}")
        partition, reverse = (
            (window_partition, window_reverse)
            if self.partition_block
            else (grid_partition, grid_reverse)
        )
        return reverse(
            self.attn(partition(x, self.rect_partition)),
            self.rect_partition,
            (height, width),
        )


def enable_rectangular(backbone, image_size=256, compartment_width=160):
    """Adapt existing modules in place, retaining every parameter/state-dict key.

    Added buffers contain geometry only and are deliberately non-persistent.
    No mutable current-shape flag, cached learned bias, or duplicate encoder.
    """
    if (
        image_size % 32
        or compartment_width % 32
        or not 0 < compartment_width < image_size
    ):
        raise ValueError("Sizes must be positive multiples of 32, with width < height")
    rect = (image_size // 32, compartment_width // 32)
    modules = [m for m in backbone.modules() if isinstance(m, PartitionAttentionCl)]
    if not modules:
        raise ValueError("Expected channels-last MaxViT partition attention")
    # Fail closed before modifying any module.
    for module in modules:
        attn = module.attn
        if type(module) is not PartitionAttentionCl or type(attn) is not AttentionCl:
            raise ValueError("Unsupported or already-adapted attention implementation")
        if not isinstance(attn.rel_pos, RelPosMlp) or attn.rel_pos.prefix_tokens:
            raise ValueError(
                "Mixed-shape encoder requires unprefixed MLP relative positions"
            )
        if tuple(module.partition_size) != (rect[0], rect[0]):
            raise ValueError("Square partition does not match configured image size")
    for module in modules:
        attn = module.attn
        rel = attn.rel_pos
        module.__class__ = MixedShapePartition
        attn.__class__ = MixedShapeAttention
        module.rect_partition = attn.rect_partition = rect
        attn.register_buffer(
            "rect_coords",
            gen_relative_log_coords(
                rect,
                rel.pretrained_window_size,
                mode=rel.mode,
                device=rel.rel_coords_log.device,
                dtype=rel.rel_coords_log.dtype,
            ),
            persistent=False,
        )
        attn.register_buffer(
            "rect_index",
            gen_relative_position_index(
                rect, device=rel.relative_position_index.device
            ).reshape(-1),
            persistent=False,
        )
    return backbone


class Net(Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rect_width = int(cfg.require("compartment_image_width"))
        self.square_size = int(cfg.image_size)
        enable_rectangular(self.backbone, self.square_size, self.rect_width)

    def _compartment_view(self, volume, *, axis, lateral):
        if volume.ndim != 5 or volume.shape[0] != 1:
            raise ValueError("Expected one (B,D,C,H,W) volume")
        depth, channels, height, width = volume.shape[1:]
        if (height, width) != (self.square_size, self.square_size):
            raise ValueError("Global input must match configured square size")
        if axis == 0:
            # Preserve original sagittal depth crop/resampling and entire in-plane FOV.
            crop = super()._compartment_view(volume, axis=axis, lateral=lateral)
        elif axis == 1:
            extent = max(1, round(width * self.compartment_fraction))
            crop = volume[..., -extent:] if lateral else volume[..., :extent]
        else:
            raise ValueError(f"Unsupported anatomical axis: {axis}")
        result = F.interpolate(
            crop.reshape(depth, channels, height, -1),
            size=(height, self.rect_width),
            mode="bilinear",
            align_corners=False,
        )
        return result.reshape(1, depth, channels, height, self.rect_width)
