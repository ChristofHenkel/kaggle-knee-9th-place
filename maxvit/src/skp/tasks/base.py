from collections import defaultdict
from typing import Any, Dict

import lightning
import torch
import torch.nn as nn
from lightning.pytorch.loggers import MLFlowLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from skp.tasks.utils import build_dataloader


class BaseTask(lightning.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.val_loss = defaultdict(list)

    def set(self, name: str, attr: Any) -> None:
        if name == "metrics":
            attr = nn.ModuleList(attr)
        setattr(self, name, attr)

    def on_train_start(self) -> None:
        required = ["model", "datasets", "optimizer", "metrics", "val_metric"]
        missing = [obj for obj in required if not hasattr(self, obj)]
        if missing:
            raise AttributeError(f"Task missing required attributes: {missing}")

        if isinstance(self.logger, MLFlowLogger):
            self.logger.log_hyperparams(self.cfg.__dict__)

    def on_train_epoch_end(self) -> None:
        if self.global_rank == 0 and isinstance(self.logger, MLFlowLogger):
            self.logger.log_metrics(
                {"training/epoch": self.current_epoch}, step=self.global_step
            )

    def on_validation_epoch_end(self) -> None:
        metrics = self.compute_validation_metrics()
        self.reset_validation_state()
        self.print_validation_metrics(metrics)
        self.log_validation_metrics(metrics)

    def compute_validation_metrics(self) -> Dict:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.compute())
        for key, values in self.val_loss.items():
            metrics[key] = torch.stack(values).mean()

        val_metric = self.val_metric
        if isinstance(val_metric, list):
            metrics["val_metric"] = torch.sum(
                torch.stack([metrics[name.lower()].cpu() for name in val_metric])
            ).item()
        else:
            metrics["val_metric"] = metrics[val_metric.lower()]
            if isinstance(metrics["val_metric"], torch.Tensor):
                metrics["val_metric"] = metrics["val_metric"].item()
        return metrics

    def reset_validation_state(self) -> None:
        self.val_loss = defaultdict(list)
        for metric in self.metrics:
            metric.reset()

    def print_validation_metrics(self, metrics: Dict) -> None:
        if self.global_rank != 0:
            return
        print("\n========")
        max_strlen = max(len(key) for key in metrics)
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"{key.ljust(max_strlen)} | {value:.4f}")

    def log_validation_metrics(self, metrics: Dict) -> None:
        if (
            self.trainer.state.stage
            == lightning.pytorch.trainer.states.RunningStage.SANITY_CHECKING
        ):
            return

        for key, value in metrics.items():
            if isinstance(self.logger, MLFlowLogger):
                self.log(
                    f"val/{key}",
                    torch.as_tensor(value, device=self.device),
                    sync_dist=True,
                )
        self.log(
            "val_metric",
            torch.as_tensor(metrics["val_metric"], device=self.device),
            sync_dist=True,
        )

    def configure_optimizers(self) -> Dict:
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None:
            return {"optimizer": self.optimizer}

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": self.cfg.get("scheduler_interval", "epoch"),
        }
        if isinstance(scheduler, ReduceLROnPlateau):
            lr_scheduler["monitor"] = "val_metric"
        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self) -> DataLoader:
        return build_dataloader(self.cfg, self.datasets[0], "train")

    def val_dataloader(self) -> DataLoader:
        return build_dataloader(self.cfg, self.datasets[1], "val")
