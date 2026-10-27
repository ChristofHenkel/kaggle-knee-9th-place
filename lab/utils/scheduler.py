from torch.optim.lr_scheduler import LambdaLR
from torch.optim import lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from collections.abc import Sequence
import math
import numpy as np


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles= 0.5, last_epoch= -1):
    """
    from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(cfg, optimizer, total_steps):

    if cfg.schedule == "ian_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, total_steps=cfg.epochs * cfg.optimizer_steps_per_epoch,
            max_lr=cfg.lr, init_lr=0.0, final_lr=cfg.final_lr, pct_start=cfg.warmup_frac,
        )
    elif cfg.schedule == "cosine_floor":
        steps = getattr(cfg, 'schedule_epochs', cfg.epochs) * cfg.optimizer_steps_per_epoch
        def lr_scale(step):
            progress = min(step / steps, 1.0)
            return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer, lr_scale)
    elif cfg.schedule == "parag_cosine":
        steps_per_epoch = getattr(cfg, 'optimizer_steps_per_epoch',
                                  (total_steps // cfg.batch_size) // cfg.grad_accumulation)
        steps = max(cfg.epochs * steps_per_epoch, 1)
        warm = max(int(cfg.warmup_frac * steps), 1)
        def lr_scale(step):
            if step < warm:
                return (step + 1) / warm
            progress = (step - warm) / max(steps - warm, 1)
            return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer, lr_scale)
    elif cfg.schedule == "onecycle":
        steps_per_epoch = cfg.optimizer_steps_per_epoch
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=cfg.epochs * steps_per_epoch,
            pct_start=cfg.onecycle_pct_start,
        )
    elif cfg.schedule == "steplr":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs_step * (total_steps // cfg.batch_size) // cfg.world_size,
            gamma=0.5,
        )
    elif cfg.schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) // cfg.world_size,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
        )
    elif cfg.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
        )

    elif cfg.schedule == "CosineAnnealingLR":
        T_max = int(np.ceil(0.5*total_steps))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=T_max,
                                                   eta_min=1e-8)


    else:
        scheduler = None

    return scheduler



class LinearWarmupCosineAnnealingLR(LRScheduler):

    def __init__(self, optimizer: Optimizer, total_steps: int, max_lr: float | Sequence[float], init_lr: float | Sequence[float]=0.0, final_lr: float | Sequence[float]=0.0, warmup_steps: int | None=None, pct_start: float | None=None, last_epoch: int=-1):
        if total_steps <= 0:
            raise ValueError(f'total_steps must be positive, got {total_steps}')
        if warmup_steps is not None and pct_start is not None:
            raise ValueError('Specify only one of warmup_steps or pct_start')
        if pct_start is not None and (not 0.0 <= pct_start <= 1.0):
            raise ValueError(f'pct_start must be in [0, 1], got {pct_start}')
        self.total_steps = int(total_steps)
        if warmup_steps is None:
            warmup_steps = round(self.total_steps * (pct_start or 0.0))
        self.warmup_steps = min(self.total_steps, max(0, int(warmup_steps)))
        max_lrs = self._as_group_values(optimizer, max_lr, 'max_lr')
        init_lrs = self._as_group_values(optimizer, init_lr, 'init_lr')
        final_lrs = self._as_group_values(optimizer, final_lr, 'final_lr')
        for idx, group in enumerate(optimizer.param_groups):
            group['lr'] = init_lrs[idx]
            group['initial_lr'] = init_lrs[idx]
            group['max_lr'] = max_lrs[idx]
            group['final_lr'] = final_lrs[idx]
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _as_group_values(optimizer: Optimizer, value: float | Sequence[float], name: str) -> list[float]:
        if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
            if len(value) != len(optimizer.param_groups):
                raise ValueError(f'{name} expected {len(optimizer.param_groups)} values, got {len(value)}')
            return [float(v) for v in value]
        return [float(value)] * len(optimizer.param_groups)

    def get_lr(self) -> list[float]:
        step = min(self.last_epoch, self.total_steps)
        lrs = []
        for group in self.optimizer.param_groups:
            init_lr = group['initial_lr']
            max_lr = group['max_lr']
            final_lr = group['final_lr']
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
