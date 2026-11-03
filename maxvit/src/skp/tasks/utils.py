import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from skp.configs.base import Config
from skp.tasks import samplers


def _require_webdataset():
    try:
        import webdataset as wds
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "webdataset is required when cfg.wds = True. Install it with "
            "`uv add webdataset` or set cfg.wds = False."
        ) from e
    return wds


def _is_ddp(cfg: Config) -> bool:
    args = cfg.get("args", {}) or {}
    strategy = args.get("strategy")
    if strategy is None:
        return False
    return strategy == "ddp" or str(strategy).startswith("ddp")


def training_batches_per_epoch(cfg: Config, dataset: Dataset) -> int:
    """Count the actual train batch sampler without decoding data or consuming RNG.

    Mirrors build_dataloader: DDP pads sampler indices, then each rank drops its
    incomplete training batch. Study samplers count studies, not dataset series.
    Explicit rank avoids requiring a process group before Trainer initialization.
    """
    if cfg.get("wds", False):
        batches = int(cfg.require("num_iterations_per_epoch"))
    else:
        if cfg.get("sampler"):
            sampler = getattr(samplers, cfg.sampler)(dataset=dataset, cfg=cfg)
            if _is_ddp(cfg):
                sampler = samplers.DistributedSamplerWrapper(
                    sampler, num_replicas=cfg.world_size, rank=0
                )
        elif _is_ddp(cfg):
            sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=0)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        batches = len(torch.utils.data.BatchSampler(sampler, cfg.batch_size, drop_last=True))
    limit = (cfg.get("args", {}) or {}).get("limit_train_batches", 1.0)
    if isinstance(limit, int):
        batches = min(batches, limit)
    elif limit is not None:
        batches = int(batches * limit)
    if batches <= 0:
        raise ValueError("Training loader has zero batches; cannot size LR schedule")
    return batches


def build_dataloader(cfg: Config, dataset: Dataset, mode: str) -> DataLoader:
    if cfg.get("wds", False):
        wds = _require_webdataset()
        num_workers = cfg.get("num_workers", 0) or 0
        persistent_workers = (cfg.get("persistent_workers", False) or False) and (
            num_workers > 0
        )
        if mode == "train":
            loader_params = {
                "batch_size": None,
                "num_workers": num_workers,
                "pin_memory": cfg.get("pin_memory", True) or False,
                "persistent_workers": persistent_workers,
            }
            if num_workers > 0:
                loader_params["prefetch_factor"] = 4
            loader = wds.WebLoader(dataset, **loader_params).with_epoch(
                cfg.num_iterations_per_epoch
            )
        else:
            val_num_workers = cfg.get("val_num_workers") or num_workers
            if cfg.get("num_val_batches_per_gpu") is not None:
                loader = wds.WebLoader(
                    dataset,
                    batch_size=None,
                    num_workers=val_num_workers,
                ).with_epoch(cfg.num_val_batches_per_gpu)
            else:
                loader = wds.WebLoader(
                    dataset,
                    batch_size=None,
                    num_workers=val_num_workers,
                )
        return loader

    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(torch.initial_seed() % 2**32)

    dataloader_params = {}
    dataloader_params["num_workers"] = cfg.num_workers
    dataloader_params["drop_last"] = mode == "train"
    dataloader_params["shuffle"] = mode == "train"
    dataloader_params["pin_memory"] = cfg.get("pin_memory", False) or False
    dataloader_params["collate_fn"] = dataset.collate_fn
    if cfg.num_workers > 0:
        dataloader_params["persistent_workers"] = (
            cfg.get("persistent_workers", False) or False
        )

    if mode == "train":
        dataloader_params["batch_size"] = cfg.batch_size
    else:
        dataloader_params["batch_size"] = cfg.get("val_batch_size") or cfg.batch_size * 2

    sampler = None
    if cfg.get("sampler") and cfg.sampler != "" and mode == "train":
        sampler = getattr(samplers, cfg.sampler)(dataset=dataset, cfg=cfg)

    if sampler:
        dataloader_params["shuffle"] = False
        if _is_ddp(cfg):
            sampler = samplers.DistributedSamplerWrapper(sampler)
        print(f"Using sampler {sampler} for training ...")
        dataloader_params["sampler"] = sampler
    elif _is_ddp(cfg):
        dataloader_params["shuffle"] = False
        if mode == "train":
            dataloader_params["sampler"] = DistributedSampler(dataset, shuffle=True)
        else:
            # Padding DistributedSampler duplicates validation series whenever
            # dataset size is not divisible by world size, which would bias
            # per-study mean predictions.
            dataloader_params["sampler"] = samplers.DistributedEvalSampler(dataset)

    loader = DataLoader(dataset, **dataloader_params, worker_init_fn=worker_init_fn)

    return loader
