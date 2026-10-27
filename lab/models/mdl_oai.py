import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mdl_knee import build_encoder_3d, encoder_features, masked_pool, to_float


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        schema = json.loads(Path(cfg.oai_schema).read_text())
        self.sizes = [len(field["classes"]) for field in schema["fields"]]
        self.encoder, dimension = build_encoder_3d(cfg)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(dimension, sum(self.sizes)))
        if cfg.channels_last:
            self.to(memory_format=torch.channels_last_3d)

    def optimizer_parameters(self):
        return [{"params": self.encoder.parameters(), "lr": self.cfg.lr_backbone},
                {"params": self.head.parameters(), "lr": self.cfg.lr_head}]

    def forward(self, batch):
        inputs = to_float(self.cfg, batch["images"], batch["smask"])
        features = encoder_features(self.encoder, inputs)
        pooled = masked_pool(features, batch["smask"].flatten(0, 1))
        logits = self.head(pooled).float().split(self.sizes, dim=-1)
        if self.training:
            targets = batch["target"].split(self.sizes, dim=-1)
            losses = torch.stack([-(target * F.log_softmax(prediction, dim=-1)).sum(-1)
                                  for target, prediction in zip(targets, logits)], dim=-1)
            mask = batch["target_mask"]
            return {"loss": ((losses * mask).sum(-1) / mask.sum(-1)).mean()}
        return {"predictions": torch.cat([prediction.softmax(-1) for prediction in logits], dim=-1)}
