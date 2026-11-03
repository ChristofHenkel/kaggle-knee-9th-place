"""Classification task that delegates all validation reporting to cohort metrics."""

from __future__ import annotations

import torch
from lightning.pytorch.trainer.states import RunningStage

from skp.callbacks.ema_callback import EMACallback
from skp.metrics.rsna_knee import ValidationMetrics
from skp.tasks.classification import Task as ClassificationTask
from skp.tasks.rsna_knee_oof import save_validation_oof


class Task(ClassificationTask):
    """Avoid the unstratified batch-mean validation loss from the base task."""

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self.model(batch, return_loss=True)
        for metric in self.metrics:
            metric.update(out, batch)
        return out["loss"]

    def on_validation_epoch_end(self) -> None:
        # Lightning's default sanity pass contains only two arbitrary batches;
        # cohort AUCs are intentionally undefined on that subset. The model
        # forward/loss still runs, then partial metric state is discarded.
        if self.trainer.state.stage == RunningStage.SANITY_CHECKING:
            self.reset_validation_state()
            return
        metrics = self.compute_validation_metrics()
        if self.cfg.get("save_validation_oof", True):
            knee_metrics = [
                metric
                for metric in self.metrics
                if isinstance(metric, ValidationMetrics)
            ]
            if len(knee_metrics) != 1:
                raise RuntimeError(
                    "OOF export requires exactly one RSNA knee validation metric"
                )
            ema_active = any(
                isinstance(callback, EMACallback) and callback.backup is not None
                for callback in self.trainer.callbacks
            )
            path = save_validation_oof(
                knee_metrics[0],
                self.datasets[1],
                self.cfg,
                epoch=self.current_epoch,
                global_step=self.global_step,
                global_rank=self.global_rank,
                prediction_weights="ema" if ema_active else "raw",
                logged_metrics=metrics,
            )
            if path is not None:
                print(f"Saved validation OOF predictions: {path}")
        self.reset_validation_state()
        self.print_validation_metrics(metrics)
        self.log_validation_metrics(metrics)
