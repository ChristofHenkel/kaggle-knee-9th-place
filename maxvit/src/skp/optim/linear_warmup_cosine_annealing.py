import math

from collections.abc import Sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """Linear warmup followed by cosine annealing.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total number of scheduler steps.
        max_lr: Peak learning rate. May be a float or one value per param group.
        init_lr: Starting learning rate. May be a float or one value per param group.
        final_lr: Final learning rate. May be a float or one value per param group.
        warmup_steps: Number of linear warmup steps. Mutually exclusive with pct_start.
        pct_start: Fraction of total_steps used for warmup.
        last_epoch: Last scheduler step, for resume.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        max_lr: float | Sequence[float],
        init_lr: float | Sequence[float] = 0.0,
        final_lr: float | Sequence[float] = 0.0,
        warmup_steps: int | None = None,
        pct_start: float | None = None,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps is not None and pct_start is not None:
            raise ValueError("Specify only one of warmup_steps or pct_start")
        if pct_start is not None and not 0.0 <= pct_start <= 1.0:
            raise ValueError(f"pct_start must be in [0, 1], got {pct_start}")

        self.total_steps = int(total_steps)
        if warmup_steps is None:
            warmup_steps = round(self.total_steps * (pct_start or 0.0))
        self.warmup_steps = min(self.total_steps, max(0, int(warmup_steps)))

        max_lrs = self._as_group_values(optimizer, max_lr, "max_lr")
        init_lrs = self._as_group_values(optimizer, init_lr, "init_lr")
        final_lrs = self._as_group_values(optimizer, final_lr, "final_lr")

        for idx, group in enumerate(optimizer.param_groups):
            group["lr"] = init_lrs[idx]
            group["initial_lr"] = init_lrs[idx]
            group["max_lr"] = max_lrs[idx]
            group["final_lr"] = final_lrs[idx]

        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _as_group_values(
        optimizer: Optimizer, value: float | Sequence[float], name: str
    ) -> list[float]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != len(optimizer.param_groups):
                raise ValueError(
                    f"{name} expected {len(optimizer.param_groups)} values, got {len(value)}"
                )
            return [float(v) for v in value]
        return [float(value)] * len(optimizer.param_groups)

    def get_lr(self) -> list[float]:
        step = min(self.last_epoch, self.total_steps)
        lrs = []

        for group in self.optimizer.param_groups:
            init_lr = group["initial_lr"]
            max_lr = group["max_lr"]
            final_lr = group["final_lr"]

            if self.warmup_steps > 0 and step < self.warmup_steps:
                pct = step / self.warmup_steps
                lr = init_lr + pct * (max_lr - init_lr)
            else:
                decay_steps = max(1, self.total_steps - self.warmup_steps)
                pct = (step - self.warmup_steps) / decay_steps
                pct = min(1.0, max(0.0, pct))
                cosine = 0.5 * (1.0 + math.cos(math.pi * pct))
                lr = final_lr + cosine * (max_lr - final_lr)

            lrs.append(lr)

        return lrs
