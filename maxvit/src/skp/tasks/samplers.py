import numpy as np

from torch.utils.data import Dataset, Sampler, DistributedSampler
from typing import Iterator, Optional

from skp.configs.base import Config


class DistributedEvalSampler(Sampler):
    """Shard evaluation data across ranks without padding or duplication."""

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        import torch.distributed as dist

        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


# From https://github.com/catalyst-team/catalyst
class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


# From https://github.com/catalyst-team/catalyst
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Yield original sampler indices assigned to the current distributed rank."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = list(super().__iter__())
        subsampler_indexes = self.dataset
        return iter([subsampler_indexes[index] for index in indexes_of_indexes])

    def set_epoch(self, epoch: int) -> None:
        """Keep wrapped epoch-aware sampling synchronized across DDP ranks."""
        super().set_epoch(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __repr__(self) -> str:
        return f"DistributedSamplerWrapper.{self.sampler}"


class StudySeriesSampler(Sampler):
    """Choose exactly one series per study, independently each epoch.

    The same deterministic global selection is constructed on every DDP rank;
    ``DistributedSamplerWrapper`` then shuffles and shards it. Standard DDP
    padding may repeat a minimal number of selected indices when the study
    count is not divisible by the number of ranks.
    """

    def __init__(self, dataset: Dataset, cfg: Config):
        super().__init__()
        if not hasattr(dataset, "items"):
            raise AttributeError(
                f"{dataset.__class__.__name__} must expose items for study-balanced sampling"
            )
        groups: dict[str, list[int]] = {}
        for index, item in enumerate(dataset.items):
            study_uid = item.get("study_uid")
            if not isinstance(study_uid, str) or not study_uid:
                raise ValueError(f"Dataset item {index} has no valid study_uid")
            groups.setdefault(study_uid, []).append(index)
        if not groups:
            raise ValueError("StudySeriesSampler received an empty dataset")
        self.study_groups = [groups[uid] for uid in sorted(groups)]
        self.seed = int(cfg.require("seed"))
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.study_groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(self.seed + self.epoch)
        selected = [
            int(group[generator.integers(0, len(group))]) for group in self.study_groups
        ]
        generator.shuffle(selected)
        return iter(selected)

    def __repr__(self) -> str:
        return f"StudySeriesSampler(studies={len(self)})"


class StudySeriesCompartmentSampler(StudySeriesSampler):
    """Select one series per study and emit both contiguous compartment items."""

    def __init__(self, dataset, cfg):
        super().__init__(dataset, cfg)
        for i in range(0, len(dataset.items), 2):
            pair = dataset.items[i : i + 2]
            if (
                len(pair) != 2
                or [p.get("compartment_side") for p in pair] != [0, 1]
                or pair[0]["series_uid"] != pair[1]["series_uid"]
            ):
                raise ValueError("Expected contiguous medial/lateral pairs per series")
        self.study_groups = [
            [i for i in group if dataset.items[i]["compartment_side"] == 0]
            for group in self.study_groups
        ]

    def __len__(self):
        return 2 * len(self.study_groups)

    def __iter__(self):
        return iter([j for i in super().__iter__() for j in (i, i + 1)])

    def __repr__(self):
        return f"StudySeriesCompartmentSampler(studies={len(self.study_groups)},crops={len(self)})"


class StudySeriesSliceSampler(Sampler):
    """Select one series per study and stratified native slices each epoch."""

    def __init__(self, dataset: Dataset, cfg: Config):
        super().__init__()
        groups = getattr(dataset, "study_series_slice_groups", None)
        if not isinstance(groups, dict) or not groups:
            raise AttributeError(
                f"{dataset.__class__.__name__} must expose non-empty "
                "study_series_slice_groups"
            )
        self.study_groups = []
        for study_uid in sorted(groups):
            series = groups[study_uid]
            if not isinstance(series, list) or not series:
                raise ValueError(f"Study {study_uid} has no slice ranges")
            normalized = []
            for start, count in series:
                start, count = int(start), int(count)
                if start < 0 or count < 1:
                    raise ValueError(
                        f"Invalid slice range for {study_uid}: {(start, count)}"
                    )
                normalized.append((start, count))
            self.study_groups.append(normalized)
        self.slices_per_series = int(cfg.require("slices_per_series"))
        if self.slices_per_series < 1:
            raise ValueError("slices_per_series must be positive")
        self.seed = int(cfg.require("seed"))
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.study_groups) * self.slices_per_series

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(self.seed + self.epoch)
        selected = []
        for series_group in self.study_groups:
            start, count = series_group[generator.integers(0, len(series_group))]
            # Jitter one position inside each equal-width depth stratum. This
            # covers the whole knee without favoring long series.
            positions = (
                np.arange(self.slices_per_series)
                + generator.random(self.slices_per_series)
            ) / self.slices_per_series
            local = np.minimum((positions * count).astype(np.int64), count - 1)
            selected.extend((start + local).tolist())
        generator.shuffle(selected)
        return iter(selected)

    def __repr__(self) -> str:
        return (
            f"StudySeriesSliceSampler(studies={len(self.study_groups)}, "
            f"slices_per_series={self.slices_per_series})"
        )


class IterationBasedSampler(Sampler):
    """
    Define epochs based on # of iterations.
    """

    def __init__(self, dataset: Dataset, cfg: Config):
        super().__init__()
        self.len_dataset = len(dataset)
        self.total_samples_per_epoch = (
            cfg.num_iterations_per_epoch * cfg.batch_size * cfg.world_size
        )
        self.available_indices = list(range(self.len_dataset))

    def __len__(self) -> int:
        return self.total_samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        num_samples = self.total_samples_per_epoch
        iteration_indices = []
        while num_samples > len(iteration_indices):
            difference = num_samples - len(iteration_indices)
            to_sample = min(difference, len(self.available_indices))
            sampled_indices = np.random.choice(
                self.available_indices, to_sample, replace=False
            )
            iteration_indices.extend(list(sampled_indices))
            self.available_indices = list(
                set(self.available_indices) - set(iteration_indices)
            )
            if len(self.available_indices) == 0:
                self.available_indices = list(range(self.len_dataset))
        return iter(iteration_indices)

    def __repr__(self) -> str:
        return "IterationBasedSampler"


def _normalized_sampling_weights(dataset: Dataset, expected_len: int) -> np.ndarray:
    if not hasattr(dataset, "sampling_weights"):
        raise AttributeError(
            f"{dataset.__class__.__name__} must define `sampling_weights` to use a "
            "weighted sampler."
        )
    weights = np.asarray(dataset.sampling_weights, dtype=np.float64)
    if weights.shape != (expected_len,):
        raise ValueError(
            f"Expected {expected_len} sampling weights, got shape {weights.shape}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("Sampling weights must be finite.")
    if np.any(weights < 0):
        raise ValueError("Sampling weights must be nonnegative.")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("Sampling weights must sum to a positive value.")
    return weights / weight_sum


class WeightedSampler(Sampler):
    """
    Samples using individual sample weights.

    If class-weighted sampling is desired, can just assign weights to samples based on class.
    """

    def __init__(self, dataset: Dataset, cfg: Config):
        super().__init__()
        self.len_dataset = len(dataset)
        self.sampling_weights = _normalized_sampling_weights(dataset, self.len_dataset)
        if cfg.get("num_iterations_per_epoch") is not None:
            self.total_samples_per_epoch = (
                cfg.num_iterations_per_epoch * cfg.batch_size * cfg.world_size
            )
        else:
            self.total_samples_per_epoch = self.len_dataset

    def __len__(self) -> int:
        return self.total_samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        indices = np.arange(self.len_dataset)
        sampled_indices = np.random.choice(
            indices, self.total_samples_per_epoch, replace=True, p=self.sampling_weights
        )
        return iter(sampled_indices.tolist())

    def __repr__(self) -> str:
        return "WeightedSampler"


class WeightedWithDecaySampler(Sampler):
    """
    Samples using individual sample weights.

    Decays sampling weights by alpha after each epoch:
      new_wt = current_wt ** alpha

    Lower alpha = more aggressive decay
    """

    def __init__(self, dataset: Dataset, cfg: Config):
        super().__init__()
        self.len_dataset = len(dataset)
        self.sampling_weights = _normalized_sampling_weights(dataset, self.len_dataset)
        self.alpha = cfg.get("sampler_decay_alpha", 0.8) or 0.8
        if cfg.get("num_iterations_per_epoch") is not None:
            self.total_samples_per_epoch = (
                cfg.num_iterations_per_epoch * cfg.batch_size * cfg.world_size
            )
        else:
            self.total_samples_per_epoch = self.len_dataset

    def __len__(self) -> int:
        return self.total_samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        indices = np.arange(self.len_dataset)
        sampled_indices = np.random.choice(
            indices,
            self.total_samples_per_epoch,
            replace=True,
            p=self.sampling_weights,
        )
        self.sampling_weights = self.sampling_weights**self.alpha
        self.sampling_weights = self.sampling_weights / self.sampling_weights.sum()
        return iter(sampled_indices.tolist())

    def __repr__(self) -> str:
        return f"WeightedWithDecaySampler, decay_alpha={self.alpha:0.2f}"
