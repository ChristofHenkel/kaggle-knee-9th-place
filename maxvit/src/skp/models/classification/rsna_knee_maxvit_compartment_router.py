"""Shared-encoder global/medial/lateral routing for knee MRI series."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


ROUTE_NAMES = ("global", "medial", "lateral")


class Net(nn.Module):
    """Route canonicalized image-space views to anatomically matched labels.

    The three views share one 2D encoder but have independent sequence and
    attention heads.  Medial/lateral views retain 65% of the appropriate
    anatomical axis and are restored to the original network shape.  A fixed
    label-to-route table prevents the router from learning protocol shortcuts.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        input_channels = int(cfg.get("num_input_channels", 3) or 3)
        image_mean = tuple(cfg.backbone_mean)
        image_std = tuple(cfg.backbone_std)
        if len(image_mean) != input_channels or len(image_std) != input_channels:
            raise ValueError("Normalization statistics must match input channels")
        backbone_args = {
            "pretrained": cfg.pretrained,
            "num_classes": 0,
            "global_pool": "avg",
            "in_chans": input_channels,
        }
        if cfg.get("backbone_img_size", False):
            image_size = int(cfg.image_size)
            backbone_args["img_size"] = tuple(
                cfg.get("backbone_input_size", (image_size, image_size))
            )
        self.backbone = create_model(cfg.backbone, **backbone_args)
        if bool(cfg.enable_gradient_checkpointing):
            self.backbone.set_grad_checkpointing(True)
        feature_dim = int(self.backbone.num_features)
        hidden_dim = int(cfg.sequence_hidden_dim)
        bidirectional = bool(cfg.sequence_bidirectional)
        sequence_dim = hidden_dim * (2 if bidirectional else 1)
        num_classes = int(cfg.num_classes)

        self.sequences = nn.ModuleList(
            [
                nn.GRU(
                    feature_dim,
                    hidden_dim,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                for _ in ROUTE_NAMES
            ]
        )
        self.attentions = nn.ModuleList(
            [nn.Linear(sequence_dim, num_classes) for _ in ROUTE_NAMES]
        )
        self.primary_weights = nn.Parameter(
            torch.randn(len(ROUTE_NAMES), num_classes, sequence_dim)
            * (sequence_dim**-0.5)
        )
        self.primary_biases = nn.Parameter(torch.zeros(len(ROUTE_NAMES), num_classes))
        self.slice_head = nn.Linear(feature_dim, num_classes)
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.encode_chunk_size = int(cfg.encode_chunk_size)
        self.lse_temperature = float(cfg.lse_temperature)
        self.aux_loss_weight = float(cfg.aux_loss_weight)
        self.compartment_fraction = float(cfg.compartment_fraction)
        if not 0.5 < self.compartment_fraction <= 1.0:
            raise ValueError("compartment_fraction must be in (0.5, 1]")

        class_names = list(cfg.class_names)
        if len(class_names) != num_classes or len(set(class_names)) != num_classes:
            raise ValueError("class_names must uniquely match num_classes")
        medial = set(cfg.medial_route_labels)
        lateral = set(cfg.lateral_route_labels)
        unknown = (medial | lateral) - set(class_names)
        if unknown or medial & lateral:
            raise ValueError(
                "Compartment route labels must be known and mutually exclusive; "
                f"unknown={sorted(unknown)}, overlap={sorted(medial & lateral)}"
            )
        route_indices = [
            1 if label in medial else 2 if label in lateral else 0
            for label in class_names
        ]
        self.register_buffer(
            "label_route_indices", torch.tensor(route_indices, dtype=torch.long)
        )
        pf_primary_weights = cfg.get("pf_oa_primary_plane_weights")
        self.pf_oa_coronal_aux_generalized_oa = bool(
            cfg.get("pf_oa_coronal_aux_generalized_oa", False)
        )
        if pf_primary_weights is None:
            if self.pf_oa_coronal_aux_generalized_oa:
                raise ValueError(
                    "Coronal generalized-OA auxiliary supervision requires "
                    "pf_oa_primary_plane_weights"
                )
            self.pf_oa_index = None
            self.medial_oa_index = None
            self.lateral_oa_index = None
            self.register_buffer("pf_oa_primary_plane_weights", None)
        else:
            expected_planes = ["Axial", "Coronal", "Sagittal"]
            if (
                not isinstance(pf_primary_weights, dict)
                or list(pf_primary_weights) != expected_planes
                or any(
                    not isinstance(pf_primary_weights[plane], (int, float))
                    or isinstance(pf_primary_weights[plane], bool)
                    or not 0.0 <= float(pf_primary_weights[plane]) <= 1.0
                    for plane in expected_planes
                )
            ):
                raise ValueError(
                    "pf_oa_primary_plane_weights must map Axial, Coronal, and "
                    "Sagittal in that order to values in [0, 1]"
                )
            self.pf_oa_index = class_names.index("PF OA")
            self.medial_oa_index = class_names.index("Medial OA")
            self.lateral_oa_index = class_names.index("Lateral OA")
            if (
                self.pf_oa_coronal_aux_generalized_oa
                and float(pf_primary_weights["Coronal"]) != 0.0
            ):
                raise ValueError(
                    "Coronal PF OA primary weight must be zero when its auxiliary "
                    "target is generalized OA"
                )
            self.register_buffer(
                "pf_oa_primary_plane_weights",
                torch.tensor(
                    [float(pf_primary_weights[plane]) for plane in expected_planes]
                ),
            )
        self.register_buffer(
            "image_mean", torch.tensor(image_mean).view(1, input_channels, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(image_std).view(1, input_channels, 1, 1)
        )
        self.criterion = None

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        chunks = []
        for start in range(0, images.shape[0], self.encode_chunk_size):
            chunks.append(self.backbone(images[start : start + self.encode_chunk_size]))
        return torch.cat(chunks)

    def _compartment_view(
        self, volume: torch.Tensor, *, axis: int, lateral: bool
    ) -> torch.Tensor:
        """Crop a canonical B=1 volume and restore its D/H/W shape."""
        if volume.shape[0] != 1 or volume.ndim != 5:
            raise ValueError("Expected one (B,D,C,H,W) canonical volume")
        depth, height, width = volume.shape[1], volume.shape[-2], volume.shape[-1]
        if axis == 0:
            extent = max(1, round(depth * self.compartment_fraction))
            cropped = volume[:, -extent:] if lateral else volume[:, :extent]
            # Interpolate D while treating image channels as 3D channels.
            cropped = cropped.permute(0, 2, 1, 3, 4)
            restored = F.interpolate(
                cropped,
                size=(depth, height, width),
                mode="trilinear",
                align_corners=False,
            )
            return restored.permute(0, 2, 1, 3, 4)
        if axis == 1:
            extent = max(1, round(width * self.compartment_fraction))
            cropped = volume[..., -extent:] if lateral else volume[..., :extent]
            restored = F.interpolate(
                cropped.reshape(depth, *cropped.shape[2:]),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            return restored.reshape(1, depth, *restored.shape[1:])
        raise ValueError(f"position_axis must be depth=0 or column=1, got {axis}")

    def _make_views(
        self,
        x: torch.Tensor,
        position_axis: torch.Tensor,
        anatomical_plane: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if position_axis.shape != (x.shape[0],):
            raise ValueError("position_axis must have one value per series")
        medial = []
        lateral = []
        for index, axis in enumerate(position_axis.tolist()):
            volume = x[index : index + 1]
            medial.append(self._compartment_view(volume, axis=axis, lateral=False))
            lateral.append(self._compartment_view(volume, axis=axis, lateral=True))
        return x, torch.cat(medial), torch.cat(lateral)

    def _loss_contract(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target = batch["y"].float()
        mask = batch["y_mask"].bool()
        base_weights = batch.get("y_weight", mask.float()).float() * mask.float()
        primary_weights = base_weights.clone()
        auxiliary_target = target.clone()
        auxiliary_weights = base_weights.clone()
        if self.pf_oa_primary_plane_weights is not None:
            if "anatomical_plane" not in batch:
                raise KeyError(
                    "Plane-aware PF OA supervision requires anatomical_plane"
                )
            plane = batch["anatomical_plane"].long().reshape(-1)
            if plane.shape != (target.shape[0],) or torch.any(
                (plane < 0) | (plane > 2)
            ):
                raise ValueError(
                    "anatomical_plane must contain one index in [0, 2] per series"
                )
            assert self.pf_oa_index is not None
            primary_weights[:, self.pf_oa_index] *= self.pf_oa_primary_plane_weights[
                plane
            ]
            if self.pf_oa_coronal_aux_generalized_oa:
                assert self.medial_oa_index is not None
                assert self.lateral_oa_index is not None
                coronal = plane == 1
                oa_indices = torch.tensor(
                    [self.medial_oa_index, self.lateral_oa_index, self.pf_oa_index],
                    device=target.device,
                )
                auxiliary_target[coronal, self.pf_oa_index] = target[coronal][
                    :, oa_indices
                ].amax(dim=1)
        return target, primary_weights, auxiliary_target, auxiliary_weights

    def forward(
        self, batch: dict[str, torch.Tensor], return_loss: bool = False
    ) -> dict[str, torch.Tensor]:
        x = batch["x"].float().div_(255.0)
        if x.ndim != 5:
            raise ValueError(f"Expected (B,D,C,H,W), got {tuple(x.shape)}")
        if "position_axis" not in batch:
            raise KeyError("Compartment routing requires position_axis metadata")
        batch_size, depth = x.shape[:2]
        x = (x - self.image_mean.unsqueeze(1)) / self.image_std.unsqueeze(1)
        views = self._make_views(
            x,
            batch["position_axis"].long(),
            batch.get("anatomical_plane"),
        )

        route_logits = []
        route_attentions = []
        route_features = []
        global_slice_logits = None
        for route_index, (view, sequence, attention_head) in enumerate(
            zip(views, self.sequences, self.attentions, strict=True)
        ):
            features = self._encode(
                view.reshape(batch_size * depth, *view.shape[2:])
            ).reshape(batch_size, depth, -1)
            if route_index == 0:
                global_slice_logits = self.slice_head(self.dropout(features))
            sequence_features, _ = sequence(features)
            attention = attention_head(sequence_features).transpose(1, 2)
            attention = attention.softmax(dim=-1)
            bag_features = torch.einsum("bcd,bdf->bcf", attention, sequence_features)
            logits = (
                self.dropout(bag_features)
                * self.primary_weights[route_index].unsqueeze(0)
            ).sum(dim=-1) + self.primary_biases[route_index]
            route_logits.append(logits)
            route_attentions.append(attention)
            route_features.append(bag_features)

        all_route_logits = torch.stack(route_logits, dim=1)
        gather_index = self.label_route_indices.view(1, 1, -1).expand(batch_size, 1, -1)
        logits = all_route_logits.gather(1, gather_index).squeeze(1)
        assert global_slice_logits is not None
        temperature = self.lse_temperature
        aux_logits = temperature * (
            torch.logsumexp(global_slice_logits / temperature, dim=1) - math.log(depth)
        )
        routed_features = (
            torch.stack(route_features, dim=1)
            .gather(
                1,
                self.label_route_indices.view(1, 1, -1, 1).expand(
                    batch_size, 1, -1, route_features[0].shape[-1]
                ),
            )
            .squeeze(1)
        )
        out = {
            "logits": logits,
            "aux_logits": aux_logits,
            "slice_logits": global_slice_logits,
            "attention": torch.stack(route_attentions, dim=1),
            "route_logits": all_route_logits,
            "label_route_indices": self.label_route_indices,
            "series_features": routed_features.mean(dim=1),
        }
        if return_loss:
            (
                target,
                primary_weights,
                auxiliary_target,
                auxiliary_weights,
            ) = self._loss_contract(batch)
            primary_matrix = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            auxiliary_matrix = F.binary_cross_entropy_with_logits(
                aux_logits, auxiliary_target, reduction="none"
            )
            primary_loss = (primary_matrix * primary_weights).sum() / (
                primary_weights.sum().clamp_min(1.0)
            )
            auxiliary_loss = (auxiliary_matrix * auxiliary_weights).sum() / (
                auxiliary_weights.sum().clamp_min(1.0)
            )
            out.update(
                {
                    "loss": primary_loss + self.aux_loss_weight * auxiliary_loss,
                    "loss_primary": primary_loss,
                    "loss_aux": auxiliary_loss,
                }
            )
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        if loss is not None:
            raise ValueError("This model computes its dual-head loss internally")
