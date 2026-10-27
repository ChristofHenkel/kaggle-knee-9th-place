import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from .config import setup_sys_path
from .artifacts import read_pinned_csv


def preprocess_series(cfg, record, side):
    from ds_knee import physical_crop_resize, slice_plan

    with np.load(record.output_path, allow_pickle=False) as archive:
        volume = archive["volume"]
    if volume.dtype != np.uint16 or len(volume) != record.n_slices:
        raise ValueError(f"Native NPZ differs from the frozen series index: {record.output_path}")
    if side == "R":
        volume = volume[::-1] if record.slot.startswith("SAG") else volume[:, :, ::-1]
    indices, real = slice_plan(len(volume), cfg.normalization_reference_slices)
    reference = volume[indices].astype(np.float32)
    lo, hi = np.percentile(reference[real > 0], [cfg.pct_low, cfg.pct_high])
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    images = np.empty((len(volume), cfg.img_px, cfg.img_px), np.uint8)
    for index in range(len(volume)):
        image = np.clip((volume[index].astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) * 255.0
        images[index] = physical_crop_resize(cfg, image, float(record.spacing_mm)).astype(np.uint8)
    return images


def build_cache(cfg, workers):
    from ds_knee import SLOTS
    from ds_knee_triplet import read_cache_manifest

    destination = Path(cfg.cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "manifest.json").exists():
            return read_cache_manifest(cfg)
        frame = read_pinned_csv(cfg.train_df, cfg.train_df_sha256)
        series = read_pinned_csv(cfg.selected_slots_csv, cfg.selected_slots_sha256)
        from .artifacts import study_lateralities
        sides = study_lateralities(cfg)
        if set(series.StudyInstanceUID) != set(frame.StudyInstanceUID):
            raise ValueError("Frozen parent series do not cover the training studies")
        if "slot_id" not in series:
            series["slot_id"] = series.slot.map({slot: index for index, slot in enumerate(SLOTS)})
        series["n_slices"] = series.output_n_slices.astype(np.int64)
        series["z_mm"] = series.SliceThickness.fillna(3.0).clip(0.5, 8.0)
        series["start"] = series.n_slices.cumsum() - series.n_slices
        series.to_csv(destination / "series.csv", index=False)
        shape = (int(series.n_slices.sum()), cfg.img_px, cfg.img_px)
        arrays = {}
        for name, dimensions, dtype in (("images", shape, np.uint8), ("completed", (len(series),), bool)):
            path = destination / f"{name}.npy"
            arrays[name] = (np.load(path, mmap_mode="r+") if path.exists() else
                            np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=dimensions))
        pending = np.flatnonzero(~arrays["completed"])
        records = list(series.itertuples(index=False))
        reuse_records = {}
        if getattr(cfg, "reuse_cache_dir", None):
            reuse_root = Path(cfg.reuse_cache_dir)
            if hashlib.sha256((reuse_root / "manifest.json").read_bytes()).hexdigest() != cfg.reuse_cache_manifest_sha256:
                raise ValueError("Source cache manifest differs from the frozen parent cache")
            reuse_images = np.load(reuse_root / "images.npy", mmap_mode="r")
            reuse_manifest = json.loads((reuse_root / "manifest.json").read_text())
            reuse_series = read_pinned_csv(reuse_root / "series.csv", reuse_manifest["series_sha256"])
            reuse_records = {record.SeriesInstanceUID: record
                             for record in reuse_series.itertuples(index=False)}
        started = time.monotonic()
        print(f"native cache {shape}: {len(pending)} series pending", flush=True)

        def fill(index):
            record = records[index]
            source = reuse_records.get(record.SeriesInstanceUID)
            if source is None:
                images = preprocess_series(cfg, record, sides.get(record.StudyInstanceUID))
            else:
                images = reuse_images[source.start:source.start + source.n_slices]
            arrays["images"][record.start:record.start + record.n_slices] = images
            return index

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for count, index in enumerate(executor.map(fill, pending), 1):
                arrays["completed"][index] = True
                if count % 100 == 0 or count == len(pending):
                    arrays["images"].flush()
                    arrays["completed"].flush()
                    print(f"cache {count}/{len(pending)} elapsed={time.monotonic() - started:.1f}s", flush=True)
        arrays["images"].flush()
        arrays["completed"].flush()
        manifest = {"recipe": cfg.cache_recipe, "shape": shape, "dtype": "uint8",
                    "train_df_sha256": cfg.train_df_sha256, "selected_slots_sha256": cfg.selected_slots_sha256,
                    "image_metadata_sha256": cfg.image_metadata_sha256, "crop_mm": cfg.crop_mm, "img_px": cfg.img_px,
                    "normalization_reference_slices": cfg.normalization_reference_slices,
                    "normalize_laterality": cfg.normalize_laterality,
                    "use_acquisition_metadata": cfg.use_acquisition_metadata,
                    "series_sha256": hashlib.sha256((destination / "series.csv").read_bytes()).hexdigest()}
        if getattr(cfg, "reuse_cache_dir", None):
            manifest.update(reuse_cache_dir=cfg.reuse_cache_dir,
                            reuse_cache_manifest_sha256=cfg.reuse_cache_manifest_sha256)
        temporary = destination / "manifest.tmp.json"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, destination / "manifest.json")
        return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    setup_sys_path()
    cv2.setNumThreads(0)
    cfg = importlib.import_module(args.config).cfg
    from .artifacts import resolve_artifacts
    resolve_artifacts(cfg)
    print(json.dumps(build_cache(cfg, args.workers)), flush=True)


if __name__ == "__main__":
    main()
