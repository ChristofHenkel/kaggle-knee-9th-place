import torch

from lightning.pytorch.accelerators.cuda import get_nvidia_gpu_stats
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger

from skp.callbacks.utils import _print_rank_zero


class GPUStatsLogger(Callback):
    def __init__(self, log_step_interval=5):
        """
        This class is a PyTorch Lightning callback which logs GPU stats
        to MLflow using the `get_nvidia_gpu_stats` function from PyTorch Lightning.

        This callback logs all available CUDA devices.

        The reason for this setup, rather than logging GPU stats from each used
        device through trainer.device or pl_module.device on each process,
        is because the logger cannot be accessed outside of rank zero, which means
        only GPU stats from rank zero would be logged.

        In most cases we are always using every available CUDA devices. If not,
        then we are just logging some unnecessary info.

        Args:
            log_step_interval (int): Interval at which to log GPU stats. Defaults to 5.
        """
        super().__init__()
        self.log_step_interval = log_step_interval

        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.devices = [
                torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
            ]
        else:
            _print_rank_zero("CUDA is not available. GPU stats will not be logged.")

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if (
            self.cuda_available
            and trainer.global_step % self.log_step_interval == 0
            and trainer.global_rank == 0
        ):
            gpu_stats = {}
            for idx, device in enumerate(self.devices):
                stats = get_nvidia_gpu_stats(device)
                stats = {k.split()[0].replace(".", "_"): v for k, v in stats.items()}
                stats = {
                    k + "_gpu" if not k.endswith("gpu") else k: v
                    for k, v in stats.items()
                }
                stats = {f"{k}_{idx}": v for k, v in stats.items()}
                gpu_stats.update(stats)
            if isinstance(trainer.logger, MLFlowLogger):
                mlflow_stats = {
                    f"system/gpu_stats/{k}": v for k, v in gpu_stats.items()
                }
                trainer.logger.log_metrics(mlflow_stats, step=trainer.global_step)
