import argparse
import copy
import glob
import json
import math
import os
import petname
import pickle
import lightning
import random
import re
import sys
import tarfile
import tempfile
import torch
import torch.distributed

from ast import literal_eval
from importlib import import_module
from lightning.pytorch import callbacks as lightning_callbacks
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.plugins import TorchSyncBatchNorm
from lightning.pytorch.utilities import rank_zero_only
from timm.layers import convert_sync_batchnorm
from typing import Optional, Tuple

from skp.callbacks import (
    ConsoleLogCallback,
    EMACallback,
    FinalCheckpointCallback,
    GPUStatsLogger,
    MLFlowSystemMonitorCallback,
)
from skp.configs.base import Config
from skp.environment import configure_gcp_gpu_environment, load_project_dotenv
from skp.optim import get_optimizer, get_scheduler
from skp.runners import run_hyperparameter_sweep, run_progressive_resizing
from skp.tracking import format_tracking_uri_for_log, validate_mlflow_tracking


class TimmSyncBatchNorm(TorchSyncBatchNorm):
    """
    Default SyncBN plugin for Lightning does not work with latest version of timm
    EfficientNets because it uses the native PyTorch `convert_sync_batchnorm` function.

    Use this plugin instead, which uses the timm helper and should work for non-timm
    models as well.
    """

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        return convert_sync_batchnorm(model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--kfold", type=str)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--overwrite_run", action="store_true")
    parser.add_argument("--double_cv", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_num_workers", type=int, default=1)
    # this flag is to not change fold in path of pretrained weights
    # e.g. if loading weights trained on another dataset
    parser.add_argument("--keep_pretrained_weights_path", action="store_true")
    # can be used to add notes e.g., for overwritten args
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--allow_local_mlflow",
        action="store_true",
        help=(
            "Explicitly authorize local MLflow tracking for this launch. "
            "Without this flag, missing/local/unreachable tracking backends "
            "abort before training."
        ),
    )

    # Lightning Trainer arguments
    # any arguments above need to be popped before passing to get_trainer function
    # due to overwrite/unknown arguments, prefer not to use LightningCLI
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    # default to benchmark=True, sync_batchnorm=True
    parser.add_argument("--no_benchmark", action="store_true")
    parser.add_argument("--no_sync_batchnorm", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=parse_limit_batches, default=None)
    parser.add_argument("--limit_val_batches", type=parse_limit_batches, default=None)
    return parser.parse_known_args()


def parse_limit_batches(value: str) -> int | float:
    """Parse Lightning batch limits while preserving integer batch counts."""
    normalized = value.strip().lower()
    if any(marker in normalized for marker in [".", "e"]):
        parsed = float(normalized)
        if parsed > 1:
            if parsed.is_integer():
                return int(parsed)
            raise argparse.ArgumentTypeError(
                "Batch-limit floats must be in [0, 1]. Use an integer for an "
                "absolute number of batches."
            )
        return parsed
    return int(normalized)


def validate_trainer_args(args: argparse.Namespace) -> None:
    """Fail early for accelerator/DDP settings that Lightning would reject later."""
    strategy = args.strategy or "auto"
    is_ddp = strategy == "ddp" or str(strategy).startswith("ddp")
    sync_batchnorm = not args.no_sync_batchnorm

    if args.accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training was requested, but torch.cuda.is_available() is False. "
                f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}. "
                "Install a PyTorch wheel compatible with the local NVIDIA driver or "
                "run with --accelerator cpu."
            )
        available_devices = torch.cuda.device_count()
        if args.devices == -1:
            args.devices = available_devices
        devices = args.devices if args.devices is not None else 1
        if devices < 1:
            raise ValueError(
                "--devices must be a positive integer or -1 for all visible "
                "CUDA devices."
            )
        if devices > available_devices:
            raise ValueError(
                f"--devices {devices} requested, but only {available_devices} CUDA "
                "device(s) are visible."
            )
        if is_ddp and not torch.distributed.is_nccl_available():
            raise RuntimeError(
                "DDP CUDA training requires NCCL, but NCCL is unavailable."
            )
    else:
        devices = args.devices if args.devices is not None else 1
        if devices == -1:
            devices = 1
            args.devices = devices
        if devices < 1:
            raise ValueError("--devices must be a positive integer.")

    if args.accelerator != "cuda" and sync_batchnorm:
        raise ValueError(
            "SyncBatchNorm is only supported for CUDA DDP runs. Add "
            "--no_sync_batchnorm when using CPU or non-CUDA accelerators."
        )

    if sync_batchnorm and not is_ddp:
        raise ValueError("SyncBatchNorm is only supported with a DDP strategy.")


@rank_zero_only
def _print_rank_zero(msg: str) -> None:
    print(msg)


def _modify_limit_batches(args: argparse.Namespace) -> argparse.Namespace:
    # float represents percentage, int represents number of batches
    if isinstance(args.limit_train_batches, float):
        if args.limit_train_batches > 1:
            args.limit_train_batches = int(args.limit_train_batches)

    if isinstance(args.limit_val_batches, float):
        if args.limit_val_batches > 1:
            args.limit_val_batches = int(args.limit_val_batches)

    return args


def load_config(args, overwrite_args):
    args = _modify_limit_batches(args)

    cfg_file = args.__dict__.pop("config")
    cfg = import_module(f"skp.configs.{cfg_file}").cfg
    cfg.args = args.__dict__
    cfg.config = cfg_file

    # overwrite config parameters, if specified in command line
    if len(overwrite_args) > 1:
        # assumes all overwrite args are prepended with '--''
        # overwrite_args will be a list following:
        # [arg_name, arg_value, arg_name, arg_value, ...]
        # here we turn it into a dict following: {arg_name: arg_value}
        overwrite_args = {
            k.replace("-", ""): v
            for k, v in zip(overwrite_args[::2], overwrite_args[1::2])
        }

    # some config parameters, such as loss_params or optimizer_params
    # are dictionaries
    # to overwrite these params via CLI, we use dot concatenation to specify
    # params within these dictionaries i.e., optimizer_params.lr
    for key in overwrite_args:
        # require that you can only overwrite an existing config parameter
        # split by . to deal with subparams
        if key in {"image_height", "image_width"}:
            raise Exception(
                f"`{key}` should not be overwritten by CLI as it is most likely referenced in other config parameters"
            )
        if overwrite_args[key].endswith(".json"):
            # for parameters (e.g., cfg.loss_params) which may have complicated
            # nested dictionary structures, if wanting to iterate over multiple
            # parameters, easier to generate a json file for each set
            _print_rank_zero(
                f"overwriting cfg.{key} with parameters in {overwrite_args[key]}"
            )
            with open(overwrite_args[key], "r") as f:
                cfg.__dict__[key] = json.load(f)
            continue
        if key.split(".")[0] in cfg.__dict__:
            if len(key.split(".")) == 1:
                # no subparam, just overwrite
                _print_rank_zero(
                    f"overwriting cfg.{key}: {cfg.__dict__[key]} -> {overwrite_args[key]}"
                )
            else:
                # subparam, need to identify dict param and key-value pair
                param_dict, param_key = key.split(".")
                _print_rank_zero(
                    f"overwriting cfg.{key}: {cfg.__dict__[param_dict][param_key]} -> {overwrite_args[key]}"
                )

            cfg_type = type(cfg.__dict__[key.split(".")[0]])
            # check if param is a dict
            if cfg_type is dict:
                assert len(key.split(".")) > 1
                param_dict, param_key = key.split(".")
                param_type = type(cfg.__dict__[param_dict][param_key])
                if param_type is bool:
                    # note that because we are not using argparse to add arguments
                    # we cannot have `store_true` args, so if it is a boolean, we need
                    # to specify True after the arg flag in command line
                    cfg.__dict__[param_dict][param_key] = overwrite_args[key] == "True"
                elif param_type is None:
                    cfg.__dict__[param_dict][param_key] = overwrite_args[key]
                else:
                    cfg.__dict__[param_dict][param_key] = param_type(
                        overwrite_args[key]
                    )
            # check if param is a list or tuple
            elif cfg_type is list or cfg_type is tuple:
                cfg.__dict__[key] = literal_eval(overwrite_args[key])
            else:
                if cfg_type is bool:
                    cfg.__dict__[key] = overwrite_args[key] == "True"
                elif cfg_type is None:
                    cfg.__dict__[key] = overwrite_args[key]
                else:
                    val = cfg_type(overwrite_args[key])
                    cfg.__dict__[key] = val
                    if key == "batch_size" and cfg.__dict__["val_batch_size"] > val:
                        # also overwrite val_batch_size if original val_batch_size is greater than overwritten batch_size
                        # if using gradient checkpointing, GPU mem during val is often similar to that of train
                        # so if reducing batch_size a lot and cfg.val_batch_size = cfg.batch_size, need to overwrite both
                        _print_rank_zero(
                            f"overwriting cfg.val_batch_size: {cfg.__dict__['val_batch_size']} -> {overwrite_args[key]}"
                        )
                        cfg.__dict__["val_batch_size"] = val
        else:
            raise Exception(f"{key} is not specified in config")

    return cfg


@rank_zero_only
def symlink_best_model_path(trainer: lightning.Trainer) -> None:
    best_model_path = None
    for callback in trainer.callbacks:
        if isinstance(callback, lightning_callbacks.ModelCheckpoint):
            best_model_path = os.path.abspath(callback.best_model_path)
            break

    if best_model_path:
        symlink_path = os.path.join(os.path.dirname(best_model_path), "best.ckpt")
        if os.path.exists(symlink_path):
            _ = os.system(f"rm {symlink_path}")
        _ = os.system(f"ln -s {best_model_path} {symlink_path}")


def generate_random_run_id(num_chars: int = 8) -> str:
    """Return a human-readable run ID in <word>-<word>-<4digit_int> format."""
    del num_chars  # Preserved for callers from older code paths.
    name = petname.Generate(words=2, separator="-", letters=10).lower()
    suffix = random.SystemRandom().randint(0, 9999)
    return f"{name}-{suffix:04d}"


def get_split_save_name(cfg: Config) -> str:
    """Return the per-split output directory name for a training run."""
    fold = cfg.get("fold")
    if fold is not None:
        return f"fold{fold}"
    return "fixed_split"


def get_mlflow_run_name(cfg: Config) -> str:
    """Return a compact run name that identifies the producing config."""
    return f"{cfg.config}/{cfg.run_id}"


def generate_experiment_save_dir(cfg: Config, run_id: Optional[str] = None) -> Config:
    save_dir = os.path.abspath(cfg.save_dir)

    if run_id:
        cfg.run_id = run_id
        _print_rank_zero(f"Using manually specified run ID: {run_id}")
    else:
        cfg.run_id = generate_random_run_id()

    save_dir = os.path.join(save_dir, cfg.config, cfg.run_id)
    cfg.save_dir = save_dir

    _print_rank_zero(f"\nRun ID: {cfg.run_id}\n")
    _print_rank_zero(f"Saving experiments to {cfg.save_dir} ...\n")

    return cfg


@rank_zero_only
def create_save_dirs(save_dir: str, overwrite: bool = False) -> None:
    os.makedirs(save_dir, exist_ok=overwrite)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=overwrite)


def get_trainer(cfg: Config) -> Tuple[lightning.Trainer, Config]:
    configured_tracking_uri = cfg.get("mlflow_tracking_uri") or os.environ.get(
        "MLFLOW_TRACKING_URI"
    )
    tracking_uri = validate_mlflow_tracking(
        configured_tracking_uri,
        allow_local=cfg.get("allow_local_mlflow", False) or False,
        save_dir=cfg.save_dir,
        dry_run=cfg.get("dry_run", False) or False,
    )
    cfg.resolved_mlflow_tracking_uri = tracking_uri

    save_dir = os.path.join(cfg.save_dir, get_split_save_name(cfg))
    create_save_dirs(save_dir, overwrite=cfg.overwrite_run)
    callbacks = []
    if cfg.get("save_console_log", True):
        callbacks.append(
            ConsoleLogCallback(
                os.path.join(
                    save_dir,
                    cfg.get("console_log_filename", "training.log") or "training.log",
                )
            )
        )
    if cfg.get("ema") and cfg.ema["on"]:
        _print_rank_zero("\n>> Using EMA ...\n")
        callbacks.append(
            EMACallback(**{k: v for k, v in cfg.ema.items() if k != "on"})
        )
    else:
        # don't save EMA parameters if not using
        _ = cfg.__dict__.pop("ema", None)

    if cfg.get("mlflow_system_monitor", True):
        callbacks.append(MLFlowSystemMonitorCallback())

    if cfg.get("log_gpu_stats", False):
        callbacks.append(
            GPUStatsLogger(
                log_step_interval=cfg.get("gpu_stats_log_step_interval", 5) or 5
            )
        )

    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    save_weights_only = cfg.get("save_weights_only", False) or False
    callbacks.extend(
        [
            lightning_callbacks.ModelCheckpoint(
                # Set dirpath explicitly to save checkpoints in the desired folder
                # This is so that we can keep the desired directory structure and format locally
                dirpath=checkpoint_dir,
                monitor="val_metric",
                filename="{epoch:03d}-{val_metric:.4f}",
                # FinalCheckpointCallback owns last.ckpt and updates it every epoch.
                # Enabling Lightning's copy as well creates last-v1.ckpt.
                save_last=False,
                save_weights_only=save_weights_only,
                mode=cfg.val_track,
                save_top_k=cfg.get("save_top_k", 1) or 1,
            ),
            FinalCheckpointCallback(
                dirpath=checkpoint_dir,
                save_weights_only=save_weights_only,
            ),
            lightning_callbacks.LearningRateMonitor(logging_interval="step"),
            lightning_callbacks.TQDMProgressBar(refresh_rate=10),
        ]
    )

    if cfg.get("early_stopping"):
        _print_rank_zero(">> Using early stopping ...")
        early_stopping = lightning_callbacks.EarlyStopping(
            patience=cfg.get("early_stopping_patience", 5),
            monitor="val_metric",
            min_delta=cfg.get("early_stopping_min_delta", 0.0),
            verbose=cfg.get("early_stopping_verbose", False) or False,
            mode=cfg.val_track,
        )
        callbacks.append(early_stopping)

    # make copy of args dictionary for strategy
    args_dict = copy.deepcopy(cfg.args)
    args_dict["sync_batchnorm"] = not args_dict.pop("no_sync_batchnorm", False)
    args_dict["benchmark"] = not args_dict.pop("no_benchmark", False)

    if cfg.args["strategy"] == "ddp":
        strategy = lightning.pytorch.strategies.DDPStrategy(
            find_unused_parameters=cfg.get("find_unused_parameters", False) or False
        )
        plugins = [TimmSyncBatchNorm()] if args_dict["sync_batchnorm"] else None
    else:
        strategy = cfg.args["strategy"]
        plugins = None

    print(f"MLflow tracking URI: {format_tracking_uri_for_log(tracking_uri)}")

    logger = MLFlowLogger(
        experiment_name=cfg.require("project"),
        run_name=get_mlflow_run_name(cfg),
        save_dir=os.path.join(cfg.save_dir, "mlflow"),
        tracking_uri=tracking_uri,
        log_model=False,
    )

    args_dict["strategy"] = strategy

    if cfg.args["strategy"] != "ddp" and args_dict["sync_batchnorm"]:
        raise Exception("SyncBatchNorm is only supported with DDP strategy")

    trainer = lightning.Trainer(
        **args_dict,
        max_epochs=cfg.num_epochs,
        callbacks=callbacks,
        plugins=plugins,
        logger=logger,
        # easier to handle custom samplers if below is False
        # see tasks/samplers.py
        # if running DDP, uses native torch DistributedSampler as default
        use_distributed_sampler=False,
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1) or 1,
        profiler="simple" if cfg.get("profile", False) else None,
    )

    return trainer, cfg


def get_loss(cfg: Config) -> torch.nn.Module:
    module, loss = cfg.loss.split(".")
    module = import_module(f"skp.losses.{module}")
    return getattr(module, loss)(cfg.loss_params)


def get_metrics(cfg: Config):
    metrics_list = []
    for metric in cfg.metrics:
        module, metric_name = metric.rsplit(".", 1)
        module = import_module(f"skp.metrics.{module}")
        metrics_list.append(getattr(module, metric_name)(cfg))
    return metrics_list


def _count_samples(shard_path_or_paths, shard_sample_suffix):
    paths = []
    if isinstance(shard_path_or_paths, str):
        paths = glob.glob(shard_path_or_paths)
    elif isinstance(shard_path_or_paths, list):
        for p in shard_path_or_paths:
            paths.extend(glob.glob(p))
    elif isinstance(shard_path_or_paths, dict):
        for p in shard_path_or_paths.values():
            if isinstance(p, str):
                paths.extend(glob.glob(p))
            elif isinstance(p, list):
                for sub_p in p:
                    paths.extend(glob.glob(sub_p))

    if isinstance(shard_sample_suffix, list):
        shard_sample_suffix = tuple(shard_sample_suffix)

    count = 0
    for p in paths:
        try:
            with tarfile.open(p, "r") as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith(shard_sample_suffix):
                        count += 1
        except Exception as e:
            print(f"Error reading shard {p}: {e}")
    return count


def get_task(
    cfg: Config, compile_model: bool = False, compile_mode: str = "max-autotune"
) -> Tuple[lightning.LightningModule, Config]:
    model = import_module(f"skp.models.{cfg.model}").Net(cfg)
    if model.criterion is None:
        # for some models (e.g., detection),
        # easier to write loss with model
        if cfg.loss is None:
            _print_rank_zero("No loss specified in config file ...")
            _print_rank_zero(
                f"Assuming loss is calculated within the model [{cfg.model}]."
            )
        else:
            loss = get_loss(cfg)
            model.set_criterion(loss)

    if cfg.get("auto_calculate_num_val_batches_per_gpu", False):
        assert (
            cfg.wds
        ), "cfg.wds must be True for auto_calculate_num_val_batches_per_gpu"
        assert getattr(
            cfg, "shard_sample_suffix", None
        ), "cfg.shard_sample_suffix must be specified if auto_calculate_num_val_batches_per_gpu is True"
        _print_rank_zero(
            "Automatically calculating number of validation batches per GPU ..."
        )
        total_val_samples = _count_samples(cfg.val_shard_fn, cfg.shard_sample_suffix)
        _print_rank_zero(f"Total validation samples: {total_val_samples}")

        val_batch_size = cfg.get("val_batch_size") or cfg.batch_size
        total_val_batches = math.ceil(total_val_samples / val_batch_size)

        num_val_batches_per_gpu = math.ceil(total_val_batches / cfg.world_size)
        cfg.num_val_batches_per_gpu = num_val_batches_per_gpu

    ds_class = import_module(f"skp.datasets.{cfg.dataset}").Dataset
    train_dataset = ds_class(cfg, "train")
    val_dataset = ds_class(cfg, "val")

    if not cfg.wds:
        cfg.n_train, cfg.n_val = len(train_dataset), len(val_dataset)

    _print_rank_zero("\n")
    _print_rank_zero(f"TRAIN : N={cfg.n_train}")
    _print_rank_zero(f"VAL   : N={cfg.n_val}\n")
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer, train_dataset=train_dataset)
    task = import_module(f"skp.tasks.{cfg.task}").Task(cfg)

    if compile_model:
        print(f"Compiling model with torch.compile (mode={compile_mode}) ...")
        model = torch.compile(model, mode=compile_mode)

    task.set("model", model)
    task.set("datasets", [train_dataset, val_dataset])
    task.set("optimizer", optimizer)
    task.set("scheduler", scheduler)
    task.set("metrics", get_metrics(cfg))
    task.set("val_metric", cfg.val_metric)

    # config has been updated with n_train, n_val
    return task, cfg


@rank_zero_only
def save_config_as_pickle(cfg: Config) -> None:
    # Can think about a solution not using pickle at a later point
    # Mainly to save a copy of the config if it was modified by command line
    # arguments, since the original config would not be correct in that case
    # although the parameters should be correct in MLflow.
    with open(os.path.join(cfg.save_dir, "config.pkl"), "wb") as f:
        pickle.dump(cfg.__dict__, f)  # dump dict because pickle can't load Config obj

    print(f"Completed run {cfg.run_id}")
    print(f"Saved experiment to {cfg.save_dir}")


@rank_zero_only
def print_environment(cfg: Config, args: argparse.Namespace) -> None:
    print("\nENVIRONMENT\n")
    print(f"  Python {sys.version}\n")
    print(f"  torch.__version__              = {torch.__version__}")
    print(f"  torch.version.cuda             = {torch.version.cuda}")
    if args.accelerator == "cuda" and torch.cuda.is_available():
        try:
            cudnn_version = torch.backends.cudnn.version()
        except RuntimeError as e:
            cudnn_version = f"unavailable ({e})"
    else:
        cudnn_version = "not used"
    print(f"  torch.backends.cudnn.version() = {cudnn_version}\n")
    print(f"  pytorch_lightning.__version__  = {lightning.__version__}\n")
    print(
        f"  world_size={cfg.world_size}, num_nodes={args.num_nodes}, num_gpus={args.devices if args.devices else 1}, num_workers={cfg.num_workers}"
    )
    print("\n")


@rank_zero_only
def save_ema_weights(trainer: lightning.Trainer) -> None:
    ema_callback = None
    last_model_path = None
    for callback in trainer.callbacks:
        if isinstance(callback, EMACallback):
            ema_callback = callback
        if isinstance(callback, lightning_callbacks.ModelCheckpoint):
            last_model_path = os.path.abspath(callback.last_model_path)
    if ema_callback is None or ema_callback.ema is None:
        return
    if not last_model_path:
        _print_rank_zero("EMA is enabled, but no checkpoint path exists yet.")
        return

    save_dir = os.path.dirname(last_model_path)
    torch.save(
        {"state_dict": ema_callback.ema.module.state_dict()},
        os.path.join(save_dir, "ema_weights.pt"),
    )


def change_fold_in_pretrained_weights_path(cfg: Config) -> Config:
    """
    if cfg.get("fold") is None:
        return cfg

    Changes pretrained weights fold to current fold in config
    For example, if running k-fold training and wanting to load
    pretrained backbone for each fold, instead of manually
    specifying each individual fold path for each run

    Assumes all folds live in the same run_id
    """
    load_attributes = [
        "load_pretrained_model",
        "load_pretrained_backbone",
        "load_pretrained_encoder",
        "load_pretrained_decoder",
    ]
    for each_att in load_attributes:
        val = cfg.get(each_att)
        if val:
            pattern = r"/fold[0-9]+/"
            if bool(re.search(pattern, val)):
                setattr(cfg, each_att, re.sub(pattern, f"/fold{cfg.fold}/", val))

    return cfg


def after_fit(trainer: lightning.Trainer) -> None:
    """Run common post-fit bookkeeping for every training mode."""
    symlink_best_model_path(trainer)
    save_ema_weights(trainer)


def main():
    load_project_dotenv()
    # uses parse_known_args() to separate into specified args
    # in parse_args and unknown args which will be exclusively used for
    # overwriting config parameters
    args, overwrite_args = parse_args()
    if args.accelerator == "cuda":
        configure_gcp_gpu_environment()
    validate_trainer_args(args)
    kfold = args.__dict__.pop("kfold")

    cfg = load_config(args, overwrite_args)
    cfg.overwrite_run = args.__dict__.pop("overwrite_run")
    cfg.double_cv = args.__dict__.pop("double_cv")
    cfg.world_size = args.num_nodes * (args.devices if args.devices else 1)
    if cfg.get("required_world_size") is not None and cfg.world_size != cfg.required_world_size:
        raise ValueError(f"This config requires world_size={cfg.required_world_size}, got {cfg.world_size}")

    notes = args.__dict__.pop("notes")
    if notes is not None:
        cfg.notes = notes

    debug_mode = args.__dict__.pop("debug")
    debug_num_workers = args.__dict__.pop("debug_num_workers")
    if debug_mode:
        cfg.debug = True
        cfg.num_workers = debug_num_workers
    else:
        cfg.debug = False

    dry_run = args.__dict__.pop("dry_run", False)
    if dry_run:
        cfg.dry_run = True
        cfg.save_dir = tempfile.mkdtemp(prefix="skp_dry_run_")
        _print_rank_zero(
            f"Dry run enabled: saving outputs to temporary directory {cfg.save_dir}"
        )

    cfg.allow_local_mlflow = args.__dict__.pop("allow_local_mlflow", False)
    configured_tracking_uri = cfg.get("mlflow_tracking_uri") or os.environ.get(
        "MLFLOW_TRACKING_URI"
    )
    preflight_tracking_uri = validate_mlflow_tracking(
        configured_tracking_uri,
        allow_local=cfg.allow_local_mlflow,
        save_dir=cfg.save_dir,
        dry_run=cfg.get("dry_run", False) or False,
    )
    _print_rank_zero(
        "MLflow preflight passed: "
        f"{format_tracking_uri_for_log(preflight_tracking_uri)}"
    )

    cfg.profile = args.__dict__.pop("profile", False)
    if cfg.profile:
        _print_rank_zero("Profiling enabled: limiting training to 50 steps.")
        cfg.num_epochs = 1
        cfg.num_iterations_per_epoch = 50
        # cfg.args["limit_train_batches"] = 50
        cfg.args["limit_val_batches"] = 0

    cfg = generate_experiment_save_dir(cfg, run_id=args.__dict__.pop("run_id"))
    print_environment(cfg, args)

    torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision", "high") or "high")

    folds = [int(_) for _ in kfold.split(",")] if kfold else [cfg.get("fold")]
    if cfg.get("double_cv") is not None:
        _print_rank_zero(
            f"Running double (nested) cross-validation with outer split {cfg.double_cv} ..."
        )

    keep_pretrained_weights_path = args.__dict__.pop(
        "keep_pretrained_weights_path", False
    )

    compile_model = args.__dict__.pop("compile", False)
    compile_mode = args.__dict__.pop("compile_mode", "max-autotune")
    for fold in folds:
        if kfold:
            _print_rank_zero(f"Running k-fold {kfold}, fold {fold} ...\n")
        elif fold is not None:
            _print_rank_zero(f"Running fold {fold} ...\n")
        else:
            _print_rank_zero("Running fixed train/val/test split ...\n")
        if fold is not None:
            cfg.fold = fold
        seed = int(cfg.get("seed", 0)) + (int(fold) if fold is not None else 0)
        lightning.seed_everything(seed, workers=True)
        if not keep_pretrained_weights_path:
            cfg = change_fold_in_pretrained_weights_path(cfg)
        if cfg.get("progressive_resizing") is not None:
            run_progressive_resizing(
                cfg,
                get_task=get_task,
                get_trainer=get_trainer,
                after_fit=after_fit,
                print_fn=_print_rank_zero,
            )
            continue
        if cfg.get("hyperparameter_sweep") is not None:
            run_hyperparameter_sweep(
                cfg,
                get_task=get_task,
                get_trainer=get_trainer,
                after_fit=after_fit,
                generate_run_id=generate_random_run_id,
                print_fn=_print_rank_zero,
            )
            continue
        task, cfg = get_task(
            cfg, compile_model=compile_model, compile_mode=compile_mode
        )
        trainer, cfg = get_trainer(cfg)
        trainer.fit(task)
        after_fit(trainer)

        if (
            cfg.get("profile", False)
            and isinstance(trainer.logger, MLFlowLogger)
            and trainer.global_rank == 0
        ):
            profiler_summary = trainer.profiler.summary()
            profiler_file = os.path.join(cfg.save_dir, "profiler_summary.txt")
            with open(profiler_file, "w") as f:
                f.write(profiler_summary)
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, profiler_file)

        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_backend() == "nccl"
        ):
            torch.distributed.barrier(device_ids=[trainer.strategy.root_device.index])
        else:
            trainer.strategy.barrier()

    # Lightning can leave the torchrun worker group alive after a completed DDP
    # fit, which blocks interpreter shutdown and any sequential experiment queue.
    # All folds have crossed the explicit barrier above, so release NCCL here.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    save_config_as_pickle(cfg)


if __name__ == "__main__":
    main()
