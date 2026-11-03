from pathlib import Path

import lightning as L


class FinalCheckpointCallback(L.Callback):
    """Write the latest completed epoch to ``last.ckpt``.

    Lightning 2.6 updates ``last.ckpt`` only when another checkpoint is saved.
    With ``save_top_k=1``, that can leave it pointing at the best epoch instead
    of the latest epoch. Saving explicitly after every completed train epoch
    restores the expected distinction between ``best.ckpt`` and ``last.ckpt``
    and leaves a useful recovery checkpoint if a later epoch is interrupted.
    """

    def __init__(self, dirpath: str | Path, save_weights_only: bool = False) -> None:
        super().__init__()
        self.path = Path(dirpath) / "last.ckpt"
        self.save_weights_only = save_weights_only

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del pl_module
        trainer.save_checkpoint(
            self.path,
            weights_only=self.save_weights_only,
        )
