from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ds_knee import (CustomDataset as BaseDataset, batch_to_device, physical_crop_resize,
                     slice_plan, split_data, tr_collate_fn, val_collate_fn)
from utils.artifacts import read_pinned_csv


def prepare_data(cfg):
    from utils.artifacts import resolve_artifacts
    resolve_artifacts(cfg)
    frame = read_pinned_csv(cfg.train_df, cfg.train_df_sha256)
    read_pinned_csv(cfg.extra_slots_csv, cfg.extra_slots_sha256)
    manifest = json.loads((Path(cfg.extra_cache_dir) / "manifest.json").read_text())
    if manifest["extra_slots_sha256"] != cfg.extra_slots_sha256:
        raise ValueError("Extra-series cache does not match the frozen selection")
    return split_data(frame, cfg)


def build_extra_cache(cfg):
    destination = Path(cfg.extra_cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "manifest.json").exists():
            return json.loads((destination / "manifest.json").read_text())
        records = list(read_pinned_csv(cfg.extra_slots_csv, cfg.extra_slots_sha256).itertuples(index=False))
        records = [record._replace(output_path=str(Path(cfg.npz_dir) / record.StudyInstanceUID / (record.SeriesInstanceUID + '.npz'))) for record in records]
        shape = (len(records), cfg.cache_slices, cfg.img_px, cfg.img_px)
        images = np.lib.format.open_memmap(destination / "images.npy", mode="w+", dtype=np.uint8, shape=shape)
        smask = np.zeros(shape[:2], np.float32)
        from utils.artifacts import study_lateralities
        sides = study_lateralities(cfg)
        started = time.monotonic()

        def fill(record):
            with np.load(record.output_path) as archive:
                volume = archive["volume"]
            if sides.get(record.StudyInstanceUID) == "R":
                volume = volume[::-1] if record.slot.startswith("SAG") else volume[:, :, ::-1]
            indices, real = slice_plan(len(volume), cfg.cache_slices)
            selected = volume[indices].astype(np.float32)
            lo, hi = np.percentile(selected[real > 0], [cfg.pct_low, cfg.pct_high])
            if not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
            for depth in range(cfg.cache_slices):
                image = np.clip((selected[depth] - lo) / (hi - lo), 0.0, 1.0) * 255.0
                images[record.cache_row, depth] = physical_crop_resize(cfg, image, float(record.spacing_mm)).astype(np.uint8)
            smask[record.cache_row] = real

        with ThreadPoolExecutor(max_workers=cfg.cache_workers) as pool:
            for count, _ in enumerate(pool.map(fill, records), 1):
                if count % 100 == 0 or count == len(records):
                    print(f"extra-series cache {count}/{len(records)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        images.flush()
        np.save(destination / "smask.npy", smask)
        manifest = {"shape": shape, "extra_slots_sha256": cfg.extra_slots_sha256,
                    "crop_mm": cfg.crop_mm, "pct_low": cfg.pct_low, "pct_high": cfg.pct_high,
                    "normalize_laterality": cfg.normalize_laterality,
                    "use_acquisition_metadata": cfg.use_acquisition_metadata, "canonical_cache": cfg.cache_dir}
        temporary = destination / "manifest.tmp.json"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(destination / "manifest.json")
        return manifest


class CustomDataset(BaseDataset):
    def __init__(self, df, cfg, aug=None, mode="train"):
        super().__init__(df, cfg, aug, mode)
        cache = Path(cfg.extra_cache_dir)
        self.extra_images = np.load(cache / "images.npy", mmap_mode="r")
        self.extra_smask = np.load(cache / "smask.npy", mmap_mode="r")
        studies = pd.read_csv(Path(cfg.cache_dir) / "studies.csv").StudyInstanceUID
        study_rows = {uid: row for row, uid in enumerate(studies)}
        self.extra_rows = np.full((len(studies), 4), -1, np.int64)
        records = read_pinned_csv(cfg.extra_slots_csv, cfg.extra_slots_sha256)
        for record in records.itertuples(index=False):
            self.extra_rows[study_rows[record.StudyInstanceUID], record.extra_rank] = record.cache_row

    def load_images(self, row):
        primary, primary_mask, primary_smask = super().load_images(row)
        images = np.zeros((12, *primary.shape[1:]), dtype=primary.dtype)
        images[:8] = primary
        mask = np.concatenate((primary_mask, np.zeros(4, np.float32)))
        smask = np.concatenate((primary_smask, np.zeros((4, primary_smask.shape[1]), np.float32)))
        for rank, cache_row in enumerate(self.extra_rows[row]):
            if cache_row >= 0:
                images[8 + rank, :, 0] = self.extra_images[cache_row]
                mask[8 + rank] = 1
                smask[8 + rank] = self.extra_smask[cache_row]
        return images, mask, smask
