import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ds_knee import (
    TARGETS, augment_study, batch_to_device, coarse_dropout_3d,
    crop_study_images, split_data)
from utils.artifacts import read_pinned_csv, resolve_artifacts


EXPECTED_KEYS = {
    "imgs", "smask", "slot_names", "slot_ids", "series_uid", "n_errors"}


def prepare_data(cfg):
    resolve_artifacts(cfg)
    frame = read_pinned_csv(cfg.train_df, cfg.train_df_sha256)
    return split_data(frame, cfg)


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = mode == "train"
        self.root = Path(cfg.cached_shards_dir)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing cached_shards manifest: {manifest_path}. "
                "Run `python -m lab.utils.prepare_v31x cache --experiment "
                f"{cfg.name.removeprefix('cfg_')}`.")
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "schema_version": 1,
            "image_shape": [
                cfg.slices_per_slot, cfg.input_channels,
                cfg.img_px, cfg.img_px],
            "slots": list(cfg.slot_names),
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"cached_shards manifest differs from config: {mismatches}")
        records = {
            row["StudyInstanceUID"]: row
            for row in manifest.get("studies", [])
            if row.get("status") in ("ok", "skipped")
        }
        self.paths = [
            self.root / f"{uid}.npz" for uid in self.df.StudyInstanceUID]
        missing = [
            str(path) for uid, path in zip(
                self.df.StudyInstanceUID, self.paths)
            if uid not in records or not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"cached_shards lacks {len(missing)} requested studies; "
                f"first: {missing[0]}")
        self.sample_sizes = np.asarray([
            int(records[uid]["n_slots"])
            for uid in self.df.StudyInstanceUID], dtype=np.int16)
        self.targets = self.df[
            [f"soft_{target}" for target in TARGETS]
        ].to_numpy(np.float32, copy=True)
        self.confidence = self.df[
            [f"conf_{target}" for target in TARGETS]
        ].to_numpy(np.float32, copy=True)
        counts = self.df.fold_group.value_counts()
        self.sample_weights = self.df.fold_group.map(
            1 / counts).to_numpy(np.float64, copy=True)

    def __len__(self):
        return len(self.df)

    def load_images(self, index):
        slots = len(self.cfg.slot_names)
        images = np.zeros((
            slots, self.cfg.slices_per_slot, self.cfg.input_channels,
            self.cfg.img_px, self.cfg.img_px), dtype=np.uint8)
        mask = np.zeros(slots, dtype=np.float32)
        smask = np.zeros(
            (slots, self.cfg.slices_per_slot), dtype=np.float32)
        with np.load(self.paths[index], allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_KEYS:
                raise ValueError(
                    f"{self.paths[index]} has keys {archive.files}, "
                    f"expected {sorted(EXPECTED_KEYS)}")
            source = archive["imgs"]
            source_mask = archive["smask"]
            slot_ids = archive["slot_ids"].astype(np.int64, copy=False)
            slot_names = archive["slot_names"].astype(str)
            if source.shape[1:] != images.shape[1:]:
                raise ValueError(
                    f"{self.paths[index]} image shape differs: "
                    f"{source.shape[1:]} != {images.shape[1:]}")
            if source_mask.shape != (
                    len(source), self.cfg.slices_per_slot):
                raise ValueError(
                    f"{self.paths[index]} slice-mask shape differs")
            if len(slot_ids) != len(source) or len(np.unique(slot_ids)) != len(
                    slot_ids):
                raise ValueError(
                    f"{self.paths[index]} has invalid slot ids")
            if np.any((slot_ids < 0) | (slot_ids >= slots)):
                raise ValueError(
                    f"{self.paths[index]} has out-of-range slot ids")
            expected_names = np.asarray(self.cfg.slot_names)[slot_ids]
            if not np.array_equal(slot_names, expected_names):
                raise ValueError(
                    f"{self.paths[index]} slot names and ids disagree")
            images[slot_ids] = source
            smask[slot_ids] = source_mask
            mask[slot_ids] = 1
        return images, mask, smask

    def __getitem__(self, index):
        images, mask, smask = self.load_images(index)
        if self.train:
            images, mask = augment_study(self.cfg, images, mask)
        images = crop_study_images(
            self.cfg, images, mask, self.train)
        if self.train and getattr(
                self.cfg, "coarse_dropout_prob", 0) > 0:
            images = coarse_dropout_3d(self.cfg, images, mask)
        return {
            "images": torch.from_numpy(images),
            "mask": torch.from_numpy(mask),
            "smask": torch.from_numpy(smask),
            "target": torch.from_numpy(self.targets[index]),
            "confidence": torch.from_numpy(self.confidence[index]),
        }


tr_collate_fn = None
val_collate_fn = None
