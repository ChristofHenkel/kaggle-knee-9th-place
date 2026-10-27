import hashlib
from pathlib import Path

import numpy as np

from ds_knee_extra import (CustomDataset as CachedStudyDataset, batch_to_device,
                           prepare_data as prepare_cached_data, tr_collate_fn, val_collate_fn)


def cached_triplets(images, smask, depth, offsets):
    centers = np.round(np.linspace(0, images.shape[1] - 1, depth)).astype(np.int64)
    indices = np.clip(centers[:, None] + np.asarray(offsets), 0, images.shape[1] - 1)
    return images[:, indices, 0], smask[:, centers]


def prepare_data(cfg):
    return prepare_cached_data(cfg)


class CustomDataset(CachedStudyDataset):
    def load_images(self, row):
        images, mask, smask = super().load_images(row)
        images, smask = cached_triplets(images, smask, self.cfg.slices_per_slot, self.cfg.slice_offsets)
        return images, mask, smask
