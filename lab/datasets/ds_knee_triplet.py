import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ds_knee import CustomDataset as StudyDataset, SLOTS, TARGETS, batch_to_device, slice_plan, split_data
from ds_knee_hybrid import pick_centres


def read_cache_manifest(cfg):
    root = Path(cfg.cache_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {"recipe": cfg.cache_recipe, "selected_slots_sha256": cfg.selected_slots_sha256,
                "image_metadata_sha256": cfg.image_metadata_sha256,
                "normalize_laterality": cfg.normalize_laterality,
                "use_acquisition_metadata": cfg.use_acquisition_metadata,
                "normalization_reference_slices": cfg.normalization_reference_slices,
                "crop_mm": cfg.crop_mm, "img_px": cfg.img_px}
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("Native-slice cache differs from the configured recipe")
    if hashlib.sha256((root / "series.csv").read_bytes()).hexdigest() != manifest["series_sha256"]:
        raise ValueError("Native-slice cache series index checksum mismatch")
    return manifest


def prepare_data(cfg):
    from utils.artifacts import read_pinned_csv

    from utils.artifacts import resolve_artifacts
    resolve_artifacts(cfg)
    read_cache_manifest(cfg)
    return split_data(read_pinned_csv(cfg.train_df, cfg.train_df_sha256), cfg)


class CustomDataset(StudyDataset):
    def __init__(self, df, cfg, aug=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = mode == "train"
        root = Path(cfg.cache_dir)
        self.images = np.load(root / "images.npy", mmap_mode="r")
        series = pd.read_csv(root / "series.csv")
        lookup = {uid: list(group.itertuples(index=False))
                  for uid, group in series.groupby("StudyInstanceUID", sort=False)}
        self.study_series = [lookup[uid] for uid in self.df.StudyInstanceUID]
        self.rows = np.arange(len(self.df))
        self.targets = self.df[[f"soft_{target}" for target in TARGETS]].to_numpy(np.float32, copy=True)
        self.confidence = self.df[[f"conf_{target}" for target in TARGETS]].to_numpy(np.float32, copy=True)
        self.loss_weight = None
        if self.train and getattr(cfg, "loss_weight_csv", None):
            weights = pd.read_csv(cfg.loss_weight_csv)
            aligned = self.df[["StudyInstanceUID"]].merge(weights, on="StudyInstanceUID", how="left", validate="one_to_one")
            self.loss_weight = aligned[TARGETS].to_numpy(np.float32, copy=True)
            if not np.isfinite(self.loss_weight).all() or not ((self.loss_weight >= 0) & (self.loss_weight <= 1)).all():
                raise ValueError("Loss weights must cover every training study and lie in [0, 1]")
        counts = self.df.fold_group.value_counts()
        self.sample_weights = self.df.fold_group.map(1 / counts).to_numpy(np.float64, copy=True)

    def load_images(self, row):
        depth, pixels = self.cfg.slices_per_slot, self.cfg.img_px
        slots = len(getattr(self.cfg, "slot_names", SLOTS))
        images = np.zeros((slots, depth, 3, pixels, pixels), np.uint8)
        mask = np.zeros(slots, np.float32)
        smask = np.zeros((slots, depth), np.float32)
        for record in self.study_series[row]:
            slot = record.slot_id
            if record.n_slices >= depth:
                centers = pick_centres(record.n_slices, depth, self.train)
                smask[slot] = 1
            else:
                centers, smask[slot] = slice_plan(record.n_slices, depth)
            step = max(1, int(round(self.cfg.triplet_delta_mm / max(float(np.float32(record.z_mm)), 0.5))))
            indices = np.clip(centers[:, None] + np.array([-step, 0, step]), 0, record.n_slices - 1)
            images[slot] = self.images[record.start + indices]
            mask[slot] = 1
        return images, mask, smask


tr_collate_fn = None
val_collate_fn = None
