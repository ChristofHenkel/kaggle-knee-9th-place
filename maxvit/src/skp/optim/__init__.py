import math

from torch import nn, optim
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from skp.configs.base import Config
from skp.optim.linear_warmup_cosine_annealing import LinearWarmupCosineAnnealingLR
from skp.optim.param_grouper import ParameterGrouper


def _as_group_values(value, num_groups: int, name: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != num_groups:
            raise ValueError(f"{name} expected {num_groups} values, got {len(value)}")
        return [float(v) for v in value]
    return [float(value)] * num_groups


def get_optimizer(cfg: Config, model: nn.Module) -> Optimizer:
    # Check for Differential Learning Rate Config
    # We assume 'parameter_groups' is a list of dicts in your cfg
    # Example:
    #
    # cfg.optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
    # cfg.parameter_groups = [
    #     # Encoder: 10x smaller LR
    #     {'match': 'encoder', 'lr_scale': 0.1, 'weight_decay': 1e-4},

    #     # Heads: Full LR, No Weight Decay
    #     {'match': 'heads', 'lr_scale': 1.0, 'weight_decay': 0.0},

    #     # Low Res Head: Full LR, No Weight Decay
    #     {'match': 'low_res_head', 'lr_scale': 1.0, 'weight_decay': 0.0}
    # ]
    optimizer_params = dict(cfg.optimizer_params)

    if cfg.get("parameter_groups"):
        # Instantiate the grouper logic here using the data from config
        # (Assuming you imported ParameterGrouper class we wrote earlier)
        grouper = ParameterGrouper(cfg.parameter_groups)

        # Apply it to the model
        base_lr = optimizer_params["lr"]
        base_wd = optimizer_params["weight_decay"]

        params = grouper(model, base_lr=base_lr, base_wd=base_wd)

    else:
        # Fallback for standard training
        params = model.parameters()

    if cfg.optimizer in {"Adam", "AdamW", "NAdam", "RAdam"}:
        # Handle tuple unpacking for betas safely
        if "beta1" in optimizer_params and "beta2" in optimizer_params:
            optimizer_params["betas"] = (
                optimizer_params.pop("beta1", 0.9),
                optimizer_params.pop("beta2", 0.999),
            )

    # Note: When 'params' is a list of groups, the 'lr' inside cfg.optimizer_params
    # acts as a global default for any group that didn't specify one,
    # but our grouper specifies one for everyone, so it's safe.
    return getattr(optim, cfg.optimizer)(params=params, **optimizer_params)


def get_scheduler(cfg: Config, optimizer: Optimizer, train_dataset=None) -> LRScheduler:
    if cfg.get("scheduler") is None:
        return None

    scheduler_params = dict(cfg.get("scheduler_params") or {})

    max_step_param = "T_max" if cfg.scheduler == "CosineAnnealingLR" else "total_steps"
    if train_dataset is not None:
        from skp.tasks.utils import training_batches_per_epoch

        steps_per_epoch = training_batches_per_epoch(cfg, train_dataset)
    elif cfg.get("sampler"):
        raise ValueError("train_dataset is required to size a custom-sampler LR schedule")
    elif cfg.get("num_iterations_per_epoch") is not None:
        steps_per_epoch = cfg.num_iterations_per_epoch
    else:
        # In DDP, each optimizer step consumes batch_size * world_size examples
        # before gradient accumulation is considered.
        effective_batch_size = cfg.batch_size * cfg.world_size
        steps_per_epoch = math.ceil(cfg.n_train / effective_batch_size)

    # For scheduler_interval="step", the scheduler is stepped once after every
    # OPTIMIZER step, so steps/epoch needs to be adjusted if using gradient accumulation
    steps_per_epoch = math.ceil(
        steps_per_epoch / (cfg.get("accumulate_grad_batches", 1) or 1)
    )

    # Step schedulers advance per optimizer update; epoch schedulers per epoch.
    total = steps_per_epoch * cfg.num_epochs
    if cfg.get("scheduler_interval", "step") == "epoch":
        total = cfg.num_epochs
    if cfg.scheduler in {"LinearWarmupCosineAnnealingLR", "CosineAnnealingLR", "OneCycleLR"}:
        scheduler_params.setdefault(max_step_param, total)
        cfg.scheduler_resolved_total_steps = scheduler_params[max_step_param]
        cfg.scheduler_optimizer_steps_per_epoch = steps_per_epoch
        print(f"LR schedule: optimizer_steps/epoch={steps_per_epoch}, "
              f"{max_step_param}={scheduler_params[max_step_param]}, "
              f"interval={cfg.get('scheduler_interval', 'step')}", flush=True)

    # --- Scheduler Selection ---
    if cfg.scheduler == "LinearWarmupCosineAnnealingLR":
        optimizer_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        if "max_lr" in scheduler_params:
            max_lrs = _as_group_values(
                scheduler_params["max_lr"], len(optimizer_lrs), "scheduler_params.max_lr"
            )
            if max_lrs != optimizer_lrs:
                raise ValueError(
                    "scheduler_params['max_lr'] must match optimizer param-group LRs. "
                    f"Got max_lr={max_lrs}, optimizer_lrs={optimizer_lrs}."
                )
        else:
            scheduler_params["max_lr"] = optimizer_lrs

        scheduler_params.setdefault("init_lr", 0.0)
        scheduler_params.setdefault("final_lr", 0.0)
        return LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, **scheduler_params
        )

    else:
        # Standard PyTorch schedulers (StepLR, ReduceLROnPlateau, etc.)
        # automatically read the LRs from the optimizer.param_groups.
        # You do NOT need to manually pass max_lr for them.
        return getattr(lr_scheduler, cfg.scheduler)(
            optimizer=optimizer, **scheduler_params
        )
