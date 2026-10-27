import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ds_knee import batch_to_device, physical_crop_resize, slice_plan


def prepare_data(cfg):
    frame = pd.read_csv(cfg.train_df)
    if cfg.data_sample > 0:
        frame = frame.iloc[:cfg.data_sample].copy()
    return frame, frame, None


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        cache = Path(cfg.cache_dir)
        self.images = np.load(cache / "images.npy", mmap_mode="r")
        self.smask = np.load(cache / "smask.npy", mmap_mode="r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if row.cache_hit:
            images = self.images[row.cache_row, row.cache_slot].copy()
            smask = self.smask[row.cache_row, row.cache_slot].copy()
        else:
            with np.load(row.output_path) as archive:
                volume = archive["volume"]
            if row.laterality == "R":
                volume = volume[::-1] if row.plane == "SAG" else volume[:, :, ::-1]
            indices, smask = slice_plan(len(volume), self.cfg.cache_slices)
            selected = volume[indices].astype(np.float32)
            low, high = (float(value) for value in np.percentile(
                selected[smask > 0], [self.cfg.pct_low, self.cfg.pct_high]))
            if not np.isfinite(high) or high <= low:
                high = low + 1.0
            images = np.stack([physical_crop_resize(self.cfg,
                               np.clip((image - low) / (high - low), 0, 1) * 255,
                               float(row.spacing_mm)).astype(np.uint8) for image in selected])
        return {"images": torch.from_numpy(images[None, :, None]),
                "smask": torch.from_numpy(smask[None]),
                "mask": torch.ones(1)}


tr_collate_fn = None
val_collate_fn = None
