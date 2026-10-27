import torch
import torch.distributed as dist
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader, WeightedRandomSampler
import numpy as np
import math
import pandas as pd

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, num_samples, num_replicas, rank, seed):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = math.ceil(num_samples / num_replicas)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.num_samples * self.num_replicas,
                                    replacement=True, generator=generator)
        return iter(indices[self.rank::self.num_replicas].tolist())


class DistributedSizeBalancedSampler(Sampler):
    def __init__(self, sizes, weights, batch_size, num_replicas, rank, seed):
        self.sizes = np.asarray(sizes)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.sizes) // (self.batch_size * self.num_replicas) * self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, len(self.sizes), replacement=True, generator=generator).numpy()
        ties = torch.rand(len(indices), generator=generator).numpy()
        indices = indices[np.lexsort((ties, self.sizes[indices]))]
        count = len(indices) // self.batch_size
        strata = [indices[index * count:(index + 1) * count] for index in range(self.batch_size)]
        strata = [stratum[torch.randperm(count, generator=generator).numpy()] for stratum in strata]
        batches = np.stack(strata, axis=1)
        order = torch.randperm(count, generator=generator).numpy()
        order = order[:count // self.num_replicas * self.num_replicas]
        return iter(batches[order[self.rank::self.num_replicas]].reshape(-1).tolist())


class OrderedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        print("TOTAL SIZE", self.total_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples : self.rank * self.num_samples + self.num_samples
        ]
        print(
            "SAMPLES",
            self.rank * self.num_samples,
            self.rank * self.num_samples + self.num_samples,
        )
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_dataset(df, cfg, mode='train'):

    #modes train, val, index
    print(f"Loading {mode} dataset")

    if mode == 'train':
        dataset = get_train_dataset(df, cfg)
    elif mode == 'val':
        dataset = get_val_dataset(df, cfg)
    elif mode == 'test':
        dataset = get_test_dataset(df, cfg)
    else:
        pass
    return dataset

def get_dataloader(ds, cfg, mode='train'):

    if mode == 'train':
        dl = get_train_dataloader(ds, cfg)
    elif mode =='val':
        dl = get_val_dataloader(ds, cfg)
    elif mode =='test':
        dl = get_test_dataloader(ds, cfg)
    return dl


def get_train_dataset(train_df, cfg):

    train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    if cfg.data_sample > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(cfg.data_sample))
    return train_dataset


def get_train_dataloader(train_ds, cfg):

    if hasattr(train_ds, 'get_sampler'):
        sampler = train_ds.get_sampler(cfg)
    elif getattr(cfg, 'balanced_batches', False):
        sampler = DistributedSizeBalancedSampler(train_ds.sample_sizes, train_ds.sample_weights,
                                                  cfg.batch_size, cfg.world_size, cfg.rank, cfg.seed)
    elif cfg.distributed:
        if getattr(cfg, 'distributed_weighted_sampling', False):
            sampler = DistributedWeightedSampler(train_ds.sample_weights,
                                                 int(len(train_ds) * cfg.random_sampler_frac),
                                                 cfg.world_size, cfg.rank, cfg.seed)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                train_ds, num_replicas=cfg.world_size, rank=cfg.local_rank, shuffle=True, seed=cfg.seed
            )
    else:
        try:
            if cfg.random_sampler_frac > 0:

                num_samples = int(len(train_ds) * cfg.random_sampler_frac)
                sample_weights = train_ds.sample_weights
                sampler = WeightedRandomSampler(sample_weights, num_samples= num_samples )
            else:
                sampler = None
        except Exception:
            sampler = None

    train_dataloader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=None if getattr(cfg, 'worker_seed', None) == 'torch' else worker_init_fn,
        persistent_workers=getattr(cfg, 'persistent_workers', False) and cfg.num_workers > 0,
        prefetch_factor=getattr(cfg, 'prefetch_factor', None) if cfg.num_workers > 0 else None,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataset(val_df, cfg, allowed_targets=None):
    val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")
    return val_dataset


def get_val_dataloader(val_ds, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(
            val_ds, num_replicas=cfg.world_size, rank=cfg.local_rank
        )
    else:
        sampler = SequentialSampler(val_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=getattr(cfg, 'eval_workers', cfg.num_workers),
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
        prefetch_factor=getattr(cfg, 'prefetch_factor', None) if getattr(cfg, 'eval_workers', cfg.num_workers) > 0 else None,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_test_dataset(test_df, cfg):
    test_dataset = cfg.CustomDataset(test_df, cfg, aug=cfg.val_aug, mode="test")
    return test_dataset


def get_test_dataloader(test_ds, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(
            test_ds, num_replicas=cfg.world_size, rank=cfg.local_rank
        )
    else:
        sampler = SequentialSampler(test_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    test_dataloader = DataLoader(
        test_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"test: dataset {len(test_ds)}, dataloader {len(test_dataloader)}")
    return test_dataloader


def read_df(fn):

    if 'parquet' in fn:
        df = pd.read_parquet(fn, engine = "fastparquet")
    else:
        df = pd.read_csv(fn)
    return df

def get_data(cfg):

    if hasattr(cfg, 'prepare_data'):
        return cfg.prepare_data(cfg)

    # setup dataset
    if isinstance(cfg.train_df, list):
        cfg.train_df = cfg.train_df[cfg.fold]
    print(f"reading {cfg.train_df}")
    df = read_df(cfg.train_df)

    if cfg.test:
        test_df = read_df(cfg.test_df)
    else:
        test_df = None

    if cfg.val_df:
        val_df = read_df(cfg.val_df)
        if cfg.fold > -1:
            val_df = val_df[val_df["fold"] == cfg.fold]
            train_df = df[df["fold"] != cfg.fold]
        else:
            train_df = df
    else:
        if cfg.fold == -1:
            val_df = df[df["fold"] == 0]
        else:
            val_df = df[df["fold"] == cfg.fold]

        train_df = df[df["fold"] != cfg.fold]

    return train_df, val_df, test_df
