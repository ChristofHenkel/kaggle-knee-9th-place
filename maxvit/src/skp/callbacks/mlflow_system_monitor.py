import lightning as L
import warnings

from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.loggers import MLFlowLogger

try:
    from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
except ModuleNotFoundError:
    SystemMetricsMonitor = None


class MLFlowSystemMonitorCallback(L.Callback):
    def __init__(self, sampling_interval: int = 10, samples_before_logging: int = 1):
        super().__init__()
        self.sampling_interval = sampling_interval
        self.samples_before_logging = samples_before_logging
        self.system_monitor = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            raise MisconfigurationException(
                "MLFlowSystemMonitorCallback requires MLFlowLogger"
            )
        if SystemMetricsMonitor is None:
            warnings.warn(
                "MLflow system metrics monitoring is unavailable. "
                "Skipping system monitoring. Set cfg.mlflow_system_monitor = False "
                "to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        # Only start on rank 0 to avoid duplicate logging in DDP
        if trainer.global_rank != 0:
            return

        self.system_monitor = SystemMetricsMonitor(
            run_id=trainer.logger.run_id,
            sampling_interval=self.sampling_interval,
            samples_before_logging=self.samples_before_logging,
        )
        self.system_monitor.start()

    def teardown(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if self.system_monitor is not None:
            self.system_monitor.finish()
