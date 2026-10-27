import json
import math
import os
from pathlib import Path
import re
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOTS = ["AXI_FS", "AXI_NFS", "COR_FS", "COR_NFS_FL", "COR_NFS_ST", "SAG_FS", "SAG_NFS_FL", "SAG_NFS_ST"]
SPLIT_PLANES = ("COR", "SAG")
PLANE_CODE = {"Axial": "AXI", "Coronal": "COR", "Sagittal": "SAG"}

def derive_series_weight(df):
    sep_rx = re.compile('[_\\-.]')
    t1_rx = re.compile('\\bt1\\b|\\bt1w\\b')
    t2_rx = re.compile('\\bt2\\b|\\bt2w\\b')
    pd_rx = re.compile('\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens')
    desc = df['SeriesDescription_mode'].fillna('').astype(str)
    desc = desc.where(desc.str.strip().str.lower() != 'dummyseriesdesc!', '')
    desc = (desc + ' ' + df['SequenceName_mode'].fillna('').astype(str)).str.lower().str.replace(sep_rx, ' ', regex=True)
    tr = pd.to_numeric(df['RepetitionTime_median'], errors='coerce')
    te = pd.to_numeric(df['EchoTime_median'], errors='coerce')
    gre = df['ScanningSequence_mode'].fillna('').astype(str).str.upper().str.contains('GR')
    t1 = desc.str.contains(t1_rx)
    t2 = desc.str.contains(t2_rx)
    pdw = desc.str.contains(pd_rx)
    weight = np.where(t1 & ~t2 & ~pdw, 'T1', np.where(t2 & ~pdw, 'T2', np.where(pdw, 'PD', np.where(gre, 'GRE', np.where(tr < 800, 'T1', np.where(te > 60, 'T2', np.where(tr >= 800, 'PD', 'UNK')))))))
    return pd.Series(np.isin(weight, ('PD', 'T2')), index=df.index)

def physical_crop_resize(cfg, img, spacing_mm):
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        spacing_mm = cfg.crop_mm / max(img.shape[0], 1)
    want = max(8, int(round(cfg.crop_mm / spacing_mm)))
    h, w = img.shape
    half = want // 2
    cy, cx = (h // 2, w // 2)
    y0, y1 = (cy - half, cy - half + want)
    x0, x1 = (cx - half, cx - half + want)
    pad_t, pad_l = (max(0, -y0), max(0, -x0))
    pad_b, pad_r = (max(0, y1 - h), max(0, x1 - w))
    out = img[max(0, y0):min(y1, h), max(0, x0):min(x1, w)]
    if pad_t or pad_b or pad_l or pad_r:
        out = np.pad(out, ((pad_t, pad_b), (pad_l, pad_r)), mode='constant')
    pixels = getattr(cfg, "cache_img_px", cfg.img_px)
    interp = cv2.INTER_AREA if out.shape[0] > pixels else cv2.INTER_LINEAR
    return cv2.resize(out, (pixels, pixels), interpolation=interp)

def slice_plan(n_z, d):
    n_z = int(n_z)
    d = int(d)
    if n_z <= 0:
        return (np.zeros(d, np.int64), np.zeros(d, np.float32))
    if n_z >= d:
        idx = np.round(np.linspace(0, n_z - 1, d)).astype(np.int64)
        return (np.clip(idx, 0, n_z - 1), np.ones(d, np.float32))
    pad = d - n_z
    left = pad // 2
    idx = np.concatenate([np.zeros(left, np.int64), np.arange(n_z, dtype=np.int64), np.full(pad - left, n_z - 1, np.int64)])
    m = np.concatenate([np.zeros(left, np.float32), np.ones(n_z, np.float32), np.zeros(pad - left, np.float32)])
    return (idx, m)

def _affine_matrix(cfg, px):
    ang = np.random.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg)
    sc = 1.0 + np.random.uniform(-cfg.aug_scale, cfg.aug_scale)
    M = cv2.getRotationMatrix2D((px / 2.0, px / 2.0), ang, sc)
    M[0, 2] += np.random.uniform(-cfg.aug_shift, cfg.aug_shift) * px
    M[1, 2] += np.random.uniform(-cfg.aug_shift, cfg.aug_shift) * px
    return M

def _gamma_lut(cfg):
    g = 1.0 + np.random.uniform(-cfg.aug_gamma, cfg.aug_gamma)
    return np.clip((np.arange(256) / 255.0) ** g * 255.0, 0, 255).astype(np.uint8)

def augment_study(cfg, imgs, mask):
    s, d, ch, h, w = imgs.shape
    if cfg.slot_dropout > 0:
        live = np.flatnonzero(mask > 0)
        if len(live) > 1:
            drop = np.random.random(len(live)) < cfg.slot_dropout
            if drop.all():
                drop[np.random.randint(len(live))] = False
            for si in live[drop]:
                mask[si] = 0.0
                imgs[si] = 0
    if not cfg.aug:
        return (imgs, mask)
    for si in range(s):
        if mask[si] == 0:
            continue
        M = _affine_matrix(cfg, h)
        lut = _gamma_lut(cfg)
        for j in range(d):
            for c in range(ch):
                imgs[si, j, c] = lut[cv2.warpAffine(imgs[si, j, c], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)]
    return (imgs, mask)

def crop_study_images(cfg, images, mask, training=False):
    crop_mm = getattr(cfg, "final_crop_mm", None)
    if crop_mm is None:
        return images
    height, width = images.shape[-2:]
    size = int(round(height * crop_mm / cfg.crop_mm))
    output = np.zeros((*images.shape[:-2], cfg.img_px, cfg.img_px), dtype=images.dtype)
    interpolation = cv2.INTER_AREA if size > cfg.img_px else cv2.INTER_LINEAR
    for slot in np.flatnonzero(mask > 0):
        if training:
            y, x = np.random.randint(height - size + 1), np.random.randint(width - size + 1)
        else:
            y, x = (height - size) // 2, (width - size) // 2
        for index in np.ndindex(images.shape[1:-2]):
            key = (slot, *index)
            crop = images[key][y:y + size, x:x + size]
            output[key] = cv2.resize(crop, (cfg.img_px, cfg.img_px), interpolation=interpolation)
    return output


def coarse_dropout_3d(cfg, images, mask):
    depth, _, height, width = images.shape[1:]
    cube = tuple(min(size, limit) for size, limit in
                 zip(cfg.coarse_dropout_cube, (depth, height, width)))
    target = int(round(depth * height * width * cfg.coarse_dropout_fraction))
    for slot in np.flatnonzero(mask > 0):
        if np.random.random() >= cfg.coarse_dropout_prob:
            continue
        keep = np.ones((depth, height, width), dtype=bool)
        removed = 0
        while removed < target:
            starts = [np.random.randint(limit - size + 1)
                      for limit, size in zip(keep.shape, cube)]
            region = tuple(slice(start, start + size) for start, size in zip(starts, cube))
            removed += int(keep[region].sum())
            keep[region] = False
        images[slot] *= keep[:, None]
    return images


def build_slot_table(cfg, keep_duplicates=False):
    ser = pd.read_csv(cfg.series_csv)
    meta = pd.read_csv(cfg.npz_meta_csv)
    meta = meta.loc[meta.status.eq("converted"), [
        "StudyInstanceUID", "SeriesInstanceUID", "output_n_slices", "Rows", "Columns",
        "PixelSpacing_row", "PixelSpacing_col", "SliceThickness"]]
    df = ser.merge(meta, on=["StudyInstanceUID", "SeriesInstanceUID"], how="inner")
    df["plane"] = df.Anatomical_Plane.map(PLANE_CODE)
    df = df.loc[df.plane.notna()].copy()
    fs = df.Fat_Suppression.astype(int).eq(1)
    if getattr(cfg, 'use_acquisition_metadata', True):
        from utils.artifacts import read_image_metadata
        columns = ['SeriesInstanceUID', 'SeriesDescription_mode', 'SequenceName_mode',
                   'ScanningSequence_mode', 'RepetitionTime_median', 'EchoTime_median']
        df = df.merge(read_image_metadata(cfg)[columns], on='SeriesInstanceUID', how='left')
        fluid = derive_series_weight(df)
    else:
        fluid = df.Fluid_Sensitive.astype(int).eq(1)
    split = df.plane.isin(SPLIT_PLANES) & ~fs
    df["slot"] = df.plane + np.where(fs, "_FS", np.where(split, np.where(fluid, "_NFS_FL", "_NFS_ST"), "_NFS"))
    df = df.sort_values(["StudyInstanceUID", "slot", "output_n_slices"], ascending=[True, True, False])
    selected = df if keep_duplicates else df.groupby(["StudyInstanceUID", "slot"], as_index=False).first()
    selected["output_path"] = [str(Path(cfg.npz_dir) / study / f"{series}.npz")
                               for study, series in zip(selected.StudyInstanceUID, selected.SeriesInstanceUID)]
    selected["spacing_mm"] = selected[["PixelSpacing_row", "PixelSpacing_col"]].mean(axis=1)
    return selected


def split_data(df, cfg):
    validation = df.fold.eq(0 if cfg.fold == -1 else cfg.fold)
    training = df.fold.ne(cfg.fold)
    if getattr(cfg, 'holdout_gold', False):
        training &= df.is_gold.eq(0)
    if getattr(cfg, 'holdout_gold', False) or getattr(cfg, 'validate_all_gold', False):
        validation |= df.is_gold.eq(1)
    validation_df = df.loc[validation].copy()
    if getattr(cfg, 'validation_train_df', None):
        from utils.artifacts import read_pinned_csv
        original = read_pinned_csv(cfg.validation_train_df)
        validation_df = original.set_index('StudyInstanceUID').loc[validation_df.StudyInstanceUID].reset_index()
    return df.loc[training].copy(), validation_df, None


def prepare_data(cfg):
    from utils.artifacts import read_pinned_csv

    return split_data(read_pinned_csv(cfg.train_df), cfg)


def build_image_cache(cfg, study_ids, slot_table):
    destination = Path(cfg.cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        return
    image_px = getattr(cfg, "cache_img_px", cfg.img_px)
    shape = (len(study_ids), len(SLOTS), cfg.cache_slices, image_px, image_px)
    pixels = np.lib.format.open_memmap(destination / "images.npy", mode="w+", dtype=np.uint8, shape=shape)
    mask = np.zeros(shape[:2], dtype=np.float32)
    smask = np.zeros(shape[:3], dtype=np.float32)
    from utils.artifacts import study_lateralities
    lateralities = study_lateralities(cfg)
    lookup = {uid: {r.slot: r for r in group.itertuples()} for uid, group in slot_table.groupby("StudyInstanceUID")}
    started = time.monotonic()
    print(f"building uint8 cache {shape}: {pixels.nbytes/1e9:.1f} GB", flush=True)
    def fill(item):
        row, uid = item
        pixels[row] = 0
        for slot_index, slot in enumerate(SLOTS):
            record = lookup[uid].get(slot)
            if record is None:
                continue
            with np.load(record.output_path) as archive:
                vol = archive["volume"]
            if lateralities.get(uid) == "R":
                vol = vol[::-1] if slot.startswith("SAG") else vol[:, :, ::-1]
            indices, real = slice_plan(len(vol), cfg.cache_slices)
            sub = vol[indices].astype(np.float32)
            lo, hi = np.percentile(sub[real > 0], [cfg.pct_low, cfg.pct_high])
            if not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
            for j in range(cfg.cache_slices):
                image = np.clip((sub[j] - lo) / (hi - lo), 0.0, 1.0) * 255.0
                pixels[row, slot_index, j] = physical_crop_resize(cfg, image, float(record.spacing_mm)).astype(np.uint8)
            mask[row, slot_index] = 1
            smask[row, slot_index] = real
        return uid
    with ThreadPoolExecutor(max_workers=cfg.cache_workers) as pool:
        for count, _ in enumerate(pool.map(fill, enumerate(study_ids)), 1):
            if count % 100 == 0 or count == len(study_ids):
                print(f"image cache {count}/{len(study_ids)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    pixels.flush()
    np.save(destination / "mask.npy", mask)
    np.save(destination / "smask.npy", smask)
    pd.DataFrame({"StudyInstanceUID": study_ids}).to_csv(destination / "studies.csv", index=False)
    manifest = {"shape": shape, "crop_mm": cfg.crop_mm, "pct_low": cfg.pct_low, "pct_high": cfg.pct_high,
                "slots": SLOTS, "normalize_laterality": getattr(cfg, "normalize_laterality", True),
                "use_acquisition_metadata": getattr(cfg, "use_acquisition_metadata", True),
                "image_metadata_sha256": getattr(cfg, "image_metadata_sha256", None), "real_slice_mean": float(smask[mask > 0].sum(-1).mean())}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("image cache ready", manifest, "live slots", int(mask.sum()), flush=True)


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = mode == "train"
        cache = Path(cfg.cache_dir)
        self.images = np.load(cache / "images.npy", mmap_mode="r")
        self.mask = np.load(cache / "mask.npy", mmap_mode="r")
        self.smask = np.load(cache / "smask.npy", mmap_mode="r")
        studies = pd.read_csv(cache / "studies.csv").StudyInstanceUID.tolist()
        study_row = {uid: row for row, uid in enumerate(studies)}
        self.rows = np.array([study_row[uid] for uid in self.df.StudyInstanceUID])
        self.targets = self.df[[f"soft_{t}" for t in TARGETS]].to_numpy(np.float32, copy=True)
        self.confidence = self.df[[f"conf_{t}" for t in TARGETS]].to_numpy(np.float32, copy=True)
        self.loss_weight = None
        if self.train and getattr(cfg, "loss_weight_csv", None):
            weights = pd.read_csv(cfg.loss_weight_csv)
            aligned = self.df[["StudyInstanceUID"]].merge(weights, on="StudyInstanceUID", how="left", validate="one_to_one")
            self.loss_weight = aligned[TARGETS].to_numpy(np.float32, copy=True)
            if not np.isfinite(self.loss_weight).all() or not ((self.loss_weight >= 0) & (self.loss_weight <= 1)).all():
                raise ValueError("Loss weights must cover every training study and lie in [0, 1]")
        counts = self.df.fold_group.value_counts()
        self.sample_weights = self.df.fold_group.map(1 / counts).to_numpy(np.float64, copy=True)

    def __len__(self):
        return len(self.df)

    def load_images(self, row):
        return self.images[row][:, :, None].copy(), self.mask[row].copy(), self.smask[row].copy()

    def __getitem__(self, index):
        row = self.rows[index]
        images, mask, smask = self.load_images(row)
        if self.train:
            images, mask = augment_study(self.cfg, images, mask)
        images = crop_study_images(self.cfg, images, mask, self.train)
        if self.train and getattr(self.cfg, "coarse_dropout_prob", 0) > 0:
            images = coarse_dropout_3d(self.cfg, images, mask)
        batch = {"images": torch.from_numpy(images), "mask": torch.from_numpy(mask),
                "smask": torch.from_numpy(smask), "target": torch.from_numpy(self.targets[index]),
                "confidence": torch.from_numpy(self.confidence[index])}
        if self.loss_weight is not None:
            batch["loss_weight"] = torch.from_numpy(self.loss_weight[index])
        return batch


tr_collate_fn = None
val_collate_fn = None


def batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
