"""Two encoded compartment views; global predictions fuse pooled regional features."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from skp.models.classification.rsna_knee_maxvit_compartment_router import Net as Parent


class Net(Parent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Regional modules retain their original indices/state names. No global GRU.
        self.sequences[0] = nn.Identity()
        self.attentions[0] = nn.Identity()
        dimension = self.primary_weights.shape[-1]
        self.global_fusion = nn.Sequential(
            nn.Linear(2 * dimension, dimension), nn.GELU()
        )

    def _condition_features(self, features, batch):
        return features

    def forward(self, batch, return_loss=False):
        x = batch["x"].float() / 255.0
        if x.ndim != 5:
            raise ValueError("Expected B,D,C,H,W input")
        if "position_axis" not in batch:
            raise KeyError("Compartment routing requires position_axis")
        b, d = x.shape[:2]
        x = (x - self.image_mean.unsqueeze(1)) / self.image_std.unsqueeze(1)
        _, medial, lateral = self._make_views(
            x, batch["position_axis"].long(), batch.get("anatomical_plane")
        )
        bags, attentions, slices = [], [], []
        for index, view in [(1, medial), (2, lateral)]:
            features = self._encode(view.reshape(b * d, *view.shape[2:])).reshape(
                b, d, -1
            )
            features = self._condition_features(features, batch)
            sequence, _ = self.sequences[index](features)
            attention = self.attentions[index](sequence).transpose(1, 2).softmax(-1)
            bags.append(torch.einsum("bcd,bdf->bcf", attention, sequence))
            attentions.append(attention)
            slices.append(self.slice_head(self.dropout(features)))
        fused = self.global_fusion(torch.cat(bags, dim=-1))
        route_features = [fused, *bags]
        route_logits = torch.stack(
            [
                (self.dropout(features) * self.primary_weights[index].unsqueeze(0)).sum(
                    -1
                )
                + self.primary_biases[index]
                for index, features in enumerate(route_features)
            ],
            dim=1,
        )
        route_index = self.label_route_indices.view(1, 1, -1).expand(b, 1, -1)
        logits = route_logits.gather(1, route_index).squeeze(1)
        all_slices = torch.cat(slices, dim=1)
        temp = self.lse_temperature
        aux_routes = torch.stack(
            [
                temp * (torch.logsumexp(s / temp, dim=1) - math.log(s.shape[1]))
                for s in [all_slices, *slices]
            ],
            dim=1,
        )
        aux_logits = aux_routes.gather(1, route_index).squeeze(1)
        features = (
            torch.stack(route_features, dim=1)
            .gather(
                1,
                self.label_route_indices.view(1, 1, -1, 1).expand(
                    b, 1, -1, fused.shape[-1]
                ),
            )
            .squeeze(1)
        )
        # Diagnostic token order: medial D then lateral D. Each attention sums to1.
        attention = torch.stack(
            [
                torch.cat(attentions, dim=-1) * 0.5,
                F.pad(attentions[0], (0, d)),
                F.pad(attentions[1], (d, 0)),
            ],
            dim=1,
        )
        out = {
            "logits": logits,
            "aux_logits": aux_logits,
            "slice_logits": all_slices,
            "attention": attention,
            "route_logits": route_logits,
            "label_route_indices": self.label_route_indices,
            "series_features": features.mean(dim=1),
            "label_features": features,
        }
        if return_loss:
            target, primary_weights, auxiliary_target, auxiliary_weights = (
                self._loss_contract(batch)
            )
            primary = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            auxiliary = F.binary_cross_entropy_with_logits(
                aux_logits, auxiliary_target, reduction="none"
            )
            primary_loss = (
                primary * primary_weights
            ).sum() / primary_weights.sum().clamp_min(1.0)
            auxiliary_loss = (
                auxiliary * auxiliary_weights
            ).sum() / auxiliary_weights.sum().clamp_min(1.0)
            out.update(
                loss=primary_loss + self.aux_loss_weight * auxiliary_loss,
                loss_primary=primary_loss,
                loss_aux=auxiliary_loss,
            )
        return out
