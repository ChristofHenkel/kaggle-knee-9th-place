"""Optional training runners that sit outside the default train path."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Callable

import numpy as np

from skp.configs.base import Config
from skp.search import generate_search


TaskFactory = Callable[[Config], tuple]
TrainerFactory = Callable[[Config], tuple]
AfterFitHook = Callable[[object], None]
Printer = Callable[[str], None]
RunIdFactory = Callable[[], str]


def run_progressive_resizing(
    cfg: Config,
    get_task: TaskFactory,
    get_trainer: TrainerFactory,
    after_fit: AfterFitHook,
    print_fn: Printer = print,
) -> None:
    """Run a sequence of trainings while changing configured image sizes.

    ``cfg.progressive_resizing`` is a dictionary whose values are equal-length
    lists, for example ``{"image_height": [256, 384], "image_width": [256, 384]}``.
    Each stage trains a fresh task and initializes from the previous stage's model
    weights. This is intentionally outside the default train path because most
    experiments do not need it.
    """
    print_fn("Running progressive resizing ...\n")
    num_stages = _validate_progressive_resizing(cfg.progressive_resizing)

    model_state_dict = {}
    cfg.save_dir = os.path.join(cfg.save_dir, "prog_resize0")
    for stage_idx in range(num_stages):
        cfg.prog_resize = stage_idx
        for key, values in cfg.progressive_resizing.items():
            print_fn(f">> setting cfg.{key} to {values[stage_idx]}")
            cfg.__dict__[key] = values[stage_idx]

        task, cfg = get_task(cfg)
        if model_state_dict:
            print_fn("Loading weights from previous progressive-resize stage ...")
            task.model.load_state_dict(model_state_dict)

        if stage_idx > 0:
            cfg.save_dir = cfg.save_dir.replace(
                f"resize{stage_idx - 1}", f"resize{stage_idx}"
            )

        trainer, cfg = get_trainer(cfg)
        trainer.fit(task)
        model_state_dict = copy.deepcopy(task.model.state_dict())
        after_fit(trainer)
        trainer.strategy.barrier()


def _validate_progressive_resizing(progressive_resizing: dict) -> int:
    if not isinstance(progressive_resizing, dict) or not progressive_resizing:
        raise ValueError("cfg.progressive_resizing must be a nonempty dictionary.")

    num_stages = None
    for key, values in progressive_resizing.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"cfg.progressive_resizing['{key}'] must be a list.")
        if num_stages is None:
            num_stages = len(values)
        elif len(values) != num_stages:
            raise ValueError("All progressive-resizing parameter lists must match.")
    return num_stages


def run_hyperparameter_sweep(
    cfg: Config,
    get_task: TaskFactory,
    get_trainer: TrainerFactory,
    after_fit: AfterFitHook,
    generate_run_id: RunIdFactory,
    print_fn: Printer = print,
) -> None:
    """Run a simple built-in hyperparameter sweep.

    ``cfg.hyperparameter_sweep`` requires ``num_trials`` and may contain either
    continuous spaces such as ``{"optimizer_params__lr": {"min": 1e-5, "max":
    1e-3, "scaling": "log"}}`` or discrete spaces such as ``{"batch_size":
    {"feasible_points": [8, 16, 32]}}``. Double underscores update nested config
    dictionaries.
    """
    num_trials = cfg.hyperparameter_sweep["num_trials"]
    print_fn(f"Running hyperparameter sweep ({num_trials} trials) ...\n")
    search_space = {
        key: value
        for key, value in cfg.hyperparameter_sweep.items()
        if key != "num_trials"
    }

    np.random.seed(88)
    hyperparameters = generate_search(search_space, num_trials)
    for trial_idx, trial in enumerate(hyperparameters):
        _apply_trial_to_config(cfg, trial, print_fn)
        if trial_idx > 0:
            save_dir = Path(cfg.save_dir).parent.absolute()
            cfg.run_id = generate_run_id()
            cfg.save_dir = os.path.join(save_dir, cfg.run_id)

        task, cfg = get_task(cfg)
        trainer, cfg = get_trainer(cfg)
        trainer.fit(task)
        after_fit(trainer)


def _apply_trial_to_config(cfg: Config, trial, print_fn: Printer) -> None:
    for name in trial._fields:
        value = getattr(trial, name)
        value_print = f"{value:0.3g}" if isinstance(value, float) else f"{value}"
        print_fn(f">> setting cfg.{name} to {value_print}")
        if "__" in name:
            dict_name, key = name.split("__", 1)
            if cfg.get(dict_name) is None:
                setattr(cfg, dict_name, {})
            cfg.__dict__[dict_name][key] = value
        else:
            cfg.__dict__[name] = value
