"""Project-scoped optimizer-step EMA with inference-correct checkpoint exports."""

from pathlib import Path

import torch

from skp import train
from skp.callbacks.ema_callback import EMACallback
from skp.tasks import utils
from train_rect_router_lowram import low_ram_loader


class KneeEMA(EMACallback):
    """Validate EMA; retain raw weights for training/resume and export EMA for inference."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Accumulated microbatches must not count as extra optimizer updates.
        step = int(trainer.global_step)
        if step <= self.global_step:
            return
        self.global_step = step - 1
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is None:
            return
        raw = self.backup if self.backup is not None else pl_module.state_dict()
        checkpoint["raw_training_state_dict"] = {
            k: v.detach().cpu().clone() for k, v in raw.items()
        }
        checkpoint["state_dict"] = {
            k: v.detach().cpu().clone() for k, v in self.ema.module.state_dict().items()
        }
        checkpoint["prediction_weights"] = "ema"

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "raw_training_state_dict" in checkpoint:
            checkpoint["state_dict"] = checkpoint["raw_training_state_dict"]


def save_ema_weights(trainer):
    if not trainer.is_global_zero:
        return
    callback = next((c for c in trainer.callbacks if isinstance(c, KneeEMA)), None)
    if callback is None or callback.ema is None:
        return
    directory = next(
        c.dirpath for c in trainer.callbacks if hasattr(c, "best_model_path")
    )
    torch.save(
        {"state_dict": callback.ema.module.state_dict(), "prediction_weights": "ema"},
        Path(directory) / "ema_weights.pt",
    )


if __name__ == "__main__":
    utils.DataLoader = low_ram_loader
    train.EMACallback = KneeEMA
    train.save_ema_weights = save_ema_weights
    train.main()
