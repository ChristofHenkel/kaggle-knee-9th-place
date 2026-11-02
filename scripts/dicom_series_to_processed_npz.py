import argparse
import json
import math
import multiprocessing as mp
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

cv2.setNumThreads(0)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "assets/data/rsna-knee-abnormality-detection"
OUT_DIR = PROJECT_ROOT / "assets/data/cached_shards"
WORKERS = 16          # processes in the study loop
STUDY_LIMIT = None    # e.g. 5 for a trial; None = every study


@dataclass
class Config:
    # copied from the infer script's Config, only the fields the sections below read
    crop_mm: float = 110.0
    img_px: int = 320
    slices_per_slot: int = 32
    cache_slices: int = 32
    band_low: float = 0.0
    band_high: float = 1.0
    pct_low: float = 0.5
    pct_high: float = 99.5
    triplet_delta_mm: float = 3.0
    normalize_laterality: bool = True
    laterality_midline_mm: float = 10.0
    laterality_workers: int = 8


cfg = Config()

BASE_SLOTS = ["AXI_FS", "AXI_NFS", "COR_FS", "COR_NFS_FL", "COR_NFS_ST", "SAG_FS",
              "SAG_NFS_FL", "SAG_NFS_ST"]
EXTRA_SLOTS = [f"EXTRA_{i:02d}" for i in range(4)]
SLOTS = BASE_SLOTS + EXTRA_SLOTS
SPLIT_PLANES = ("COR", "SAG")
PLANE_CODE = {"Axial": "AXI", "Coronal": "COR", "Sagittal": "SAG"}

IMG_ROOT = INPUT_ROOT / "train_series"
SERIES_CSV = INPUT_ROOT / "train_series.csv"
_SER_SEL = None
SLOT_LOOKUP = {}
STUDY_LATERALITY = {}
SERIES_SCAN = {}


# ============================================================================================
# 1. Laterality and sequence weighting - the rules. Unchanged from the v098 family's own
# infer scripts.
# ============================================================================================

def _lat_to_str(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except TypeError:
        pass
    t = str(v)
    return t if t else None


def _lat_float_list(value, n=None):
    if value is None:
        return None
    try:
        vals = [float(x) for x in value]
    except Exception:
        try:
            vals = [float(x) for x in str(value).replace("\\", "|").split("|")]
        except Exception:
            return None
    if n is not None and len(vals) < n:
        return None
    return vals


def normalized_laterality(value):
    text = _lat_to_str(value)
    if text is None:
        return None
    text = text.strip().upper()
    if text in {"L", "LEFT"}:
        return "L"
    if text in {"R", "RIGHT"}:
        return "R"
    return None


def pixel_spacing_pair(value):
    vals = _lat_float_list(value, 2)
    if vals is None:
        return None
    row_spacing, col_spacing = float(vals[0]), float(vals[1])
    if not (np.isfinite(row_spacing) and np.isfinite(col_spacing)
            and row_spacing > 0 and col_spacing > 0):
        return None
    return row_spacing, col_spacing


def image_center_x(h):
    ipp = _lat_float_list(h.get("ImagePositionPatient"), 3)
    iop = _lat_float_list(h.get("ImageOrientationPatient"), 6)
    spacing = pixel_spacing_pair(h.get("PixelSpacing"))
    rows, cols = h.get("Rows"), h.get("Columns")
    if ipp is None or iop is None or spacing is None or rows is None or cols is None:
        return None
    row_spacing, col_spacing = spacing
    p = np.asarray(ipp[:3], dtype=np.float64)
    row_dir = np.asarray(iop[:3], dtype=np.float64)
    col_dir = np.asarray(iop[3:6], dtype=np.float64)
    center = (p + row_dir * col_spacing * (cols - 1) / 2.0
              + col_dir * row_spacing * (rows - 1) / 2.0)
    return float(center[0]) if np.isfinite(center[0]) else None


def mode_or_none(values):
    vals = [v for v in values if v is not None
            and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    return pd.Series(vals).mode(dropna=True).iloc[0]


def infer_laterality_from_headers(headers):
    tag_values = [normalized_laterality(h.get("Laterality")) for h in headers]
    tag_values = [v for v in tag_values if v is not None]
    center_values = [image_center_x(h) for h in headers]
    center_values = [v for v in center_values if v is not None and np.isfinite(v)]
    center_x = float(np.median(center_values)) if center_values else None
    if tag_values:
        return {"laterality": mode_or_none(tag_values), "source": "dicom_tag",
                "center_x": center_x}
    if center_x is None or abs(center_x) <= cfg.laterality_midline_mm:
        return {"laterality": None, "source": "unresolved", "center_x": center_x}
    return {"laterality": "R" if center_x < 0 else "L", "source": "geometry",
            "center_x": center_x}


def resolve_study(rows):
    tagged = [r["laterality"] for r in rows
              if r["source"] == "dicom_tag" and r["laterality"] is not None]
    centers = [r["center_x"] for r in rows
               if r["center_x"] is not None and np.isfinite(r["center_x"])]
    center_x = float(np.median(centers)) if centers else None
    if tagged:
        return mode_or_none(tagged), "dicom_tag"
    if center_x is not None and abs(center_x) > cfg.laterality_midline_mm:
        return ("R" if center_x < 0 else "L"), "geometry"
    return None, "unresolved"


_WEIGHT_COLS = ("SeriesDescription_mode", "SequenceName_mode", "ScanningSequence_mode",
                "RepetitionTime_median", "EchoTime_median")


def _mode_or_none(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    return pd.Series(vals).mode(dropna=True).iloc[0]


def _median_or_none(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def _acq_text(h, tag):
    v = h.get(tag)
    if v is None:
        return None
    t = str(v)
    return t if t else None


def _acq_num(h, tag):
    v = h.get(tag)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def acquisition_meta(heads):
    return {
        "SeriesDescription_mode": _mode_or_none([_acq_text(h, "SeriesDescription")
                                                 for h in heads]),
        "SequenceName_mode": _mode_or_none([_acq_text(h, "SequenceName") for h in heads]),
        "ScanningSequence_mode": _mode_or_none([_acq_text(h, "ScanningSequence")
                                                for h in heads]),
        "RepetitionTime_median": _median_or_none([_acq_num(h, "RepetitionTime")
                                                  for h in heads]),
        "EchoTime_median": _median_or_none([_acq_num(h, "EchoTime") for h in heads]),
    }


def derive_series_weight(df):
    sep_rx = re.compile(r"[_\-.]")
    t1_rx = re.compile(r"\bt1\b|\bt1w\b")
    t2_rx = re.compile(r"\bt2\b|\bt2w\b")
    pd_rx = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")
    desc = df["SeriesDescription_mode"].fillna("").astype(str)
    desc = desc.where(desc.str.strip().str.lower() != "dummyseriesdesc!", "")
    desc = ((desc + " " + df["SequenceName_mode"].fillna("").astype(str))
            .str.lower().str.replace(sep_rx, " ", regex=True))
    tr = pd.to_numeric(df["RepetitionTime_median"], errors="coerce")
    te = pd.to_numeric(df["EchoTime_median"], errors="coerce")
    gre = df["ScanningSequence_mode"].fillna("").astype(str).str.upper().str.contains("GR")
    t1 = desc.str.contains(t1_rx)
    t2 = desc.str.contains(t2_rx)
    pdw = desc.str.contains(pd_rx)
    weight = np.where(
        t1 & ~t2 & ~pdw, "T1",
        np.where(t2 & ~pdw, "T2",
                 np.where(pdw, "PD",
                          np.where(gre, "GRE",
                                   np.where(tr < 800, "T1",
                                            np.where(te > 60, "T2",
                                                     np.where(tr >= 800, "PD", "UNK")))))))
    return pd.Series(np.isin(weight, ("PD", "T2")), index=df.index)


# ============================================================================================
# 2. One header pass - scoped to _SER_SEL (this process's own shard). Unchanged from the
# v098 family.
# ============================================================================================

_HDR_TAGS = ("Laterality", "ImagePositionPatient", "ImageOrientationPatient",
             "PixelSpacing", "Rows", "Columns", "InstanceNumber",
             "SeriesDescription", "SequenceName", "ScanningSequence",
             "RepetitionTime", "EchoTime", "SliceThickness")


def natural_path_key(p):
    return str(p)


def read_header(path):
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None
    out = {"path": path}
    for tag in _HDR_TAGS:
        try:
            out[tag] = getattr(ds, tag, None)
        except Exception:
            out[tag] = None
    return out


def geo_record(path, h):
    if h is None:
        return {"path": path, "ok": 0, "k": np.nan, "instance": np.inf}
    try:
        iop = np.asarray(h["ImageOrientationPatient"], dtype=np.float64)
        ipp = np.asarray(h["ImagePositionPatient"], dtype=np.float64)
        if iop.size < 6 or ipp.size < 3:
            raise ValueError("missing IOP/IPP")
        normal = np.cross(iop[:3], iop[3:6])
        inst = h["InstanceNumber"]
        return {"path": path, "ok": 1, "k": float(np.dot(ipp[:3], normal)),
                "instance": float(np.inf if inst is None else inst)}
    except Exception:
        return {"path": path, "ok": 0, "k": np.nan, "instance": np.inf}


def order_from(records):
    need = max(2, int(0.8 * len(records)))
    geometric = [h for h in records if h["ok"] and np.isfinite(h["k"])]
    if len(geometric) >= need:
        ordered = sorted(records, key=lambda h: (h["k"] if np.isfinite(h["k"]) else np.inf,
                                                 h["instance"], natural_path_key(h["path"])))
        return [h["path"] for h in ordered], "geometric"
    numbered = [h for h in records if np.isfinite(h["instance"])]
    if len(numbered) >= need:
        ordered = sorted(records, key=lambda h: (h["instance"], natural_path_key(h["path"])))
        return [h["path"] for h in ordered], "instance_number"
    return sorted((h["path"] for h in records), key=natural_path_key), "filename_fallback"


def spacing_from(h):
    try:
        sp = h["PixelSpacing"]
        return (float(sp[0]) + float(sp[1])) / 2.0
    except Exception:
        return np.nan


def z_mm_from(h):
    try:
        v = float(h["SliceThickness"])
        if not np.isfinite(v):
            return 3.0
        return float(min(max(v, 0.5), 8.0))
    except Exception:
        return 3.0


def scan_series(series_dir):
    key = str(series_dir)
    got = SERIES_SCAN.get(key)
    if got is not None:
        return got
    d = Path(series_dir)
    paths = sorted(d.glob("*.dcm")) if d.is_dir() else []
    heads = [read_header(p) for p in paths]
    ordered, how = order_from([geo_record(p, h) for p, h in zip(paths, heads)]) \
        if paths else ([], "empty")
    by_path = {str(h["path"]): h for h in heads if h is not None}
    rec = {
        "dir": key,
        "n": len(paths),
        "ordered": ordered,
        "order": how,
        "lat": infer_laterality_from_headers([h for h in heads if h is not None]),
        "acq": acquisition_meta([h for h in heads if h is not None]),
        "spacing": spacing_from(by_path[str(ordered[0])]) if ordered
                   and str(ordered[0]) in by_path else np.nan,
        "z_mm": z_mm_from(by_path[str(ordered[0])]) if ordered
                and str(ordered[0]) in by_path else 3.0,
    }
    SERIES_SCAN[key] = rec
    return rec


def prescan(pairs, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda sv: scan_series(IMG_ROOT / sv[0] / sv[1]), pairs))


# ============================================================================================
# 3. Slot table - scoped to _SER_SEL (this process's own shard). 8 base slots + up to 4
# EXTRA_ overflow slots (ragged), unchanged from the v098 family.
# ============================================================================================

def build_slot_table():
    ser = _SER_SEL.copy()
    ser["plane"] = ser["Anatomical_Plane"].map(PLANE_CODE)
    ser = ser[ser["plane"].notna()].copy()

    scans = [scan_series(IMG_ROOT / s / v)
             for s, v in zip(ser.StudyInstanceUID, ser.SeriesInstanceUID)]
    ser["series_dir"] = [r["dir"] for r in scans]
    ser["n_slices"] = [r["n"] for r in scans]
    keep = ser.n_slices > 0
    ser = ser[keep].copy()
    scans = [r for r, k in zip(scans, keep) if k]

    fs = ser["Fat_Suppression"].astype(int) == 1
    split = ser["plane"].isin(SPLIT_PLANES) & ~fs
    for c in _WEIGHT_COLS:
        ser[c] = [r["acq"][c] for r in scans]
    fluid = pd.Series(False, index=ser.index)
    if bool(split.any()):
        fluid.loc[split] = derive_series_weight(ser.loc[split])

    ser["slot"] = ser["plane"] + np.where(
        fs, "_FS", np.where(split, np.where(fluid, "_NFS_FL", "_NFS_ST"), "_NFS"))

    ser = ser.sort_values(["StudyInstanceUID", "slot", "n_slices"],
                          ascending=[True, True, False])
    primary = ser.groupby(["StudyInstanceUID", "slot"], as_index=False).first()
    used = set(primary.SeriesInstanceUID)
    rest = ser[~ser.SeriesInstanceUID.isin(used)].sort_values(
        ["StudyInstanceUID", "n_slices", "SeriesInstanceUID"],
        ascending=[True, False, True]).copy()
    rest["extra_rank"] = rest.groupby("StudyInstanceUID").cumcount()
    rest = rest[rest.extra_rank < len(EXTRA_SLOTS)].copy()
    rest["slot"] = "EXTRA_" + rest.extra_rank.astype(str).str.zfill(2)
    return pd.concat([primary, rest], ignore_index=True)


def build_study_laterality():
    if not cfg.normalize_laterality:
        return {}
    per_study = {}
    for s, v in zip(_SER_SEL.StudyInstanceUID, _SER_SEL.SeriesInstanceUID):
        per_study.setdefault(s, []).append(scan_series(IMG_ROOT / s / v)["lat"])
    return {s: resolve_study(rows)[0] for s, rows in per_study.items()}


def flip_for(uid, slot, study_laterality):
    if not cfg.normalize_laterality or study_laterality.get(uid) != "R":
        return False, False
    p = slot[:3]
    return p in ("AXI", "COR"), p == "SAG"


# ============================================================================================
# 4. Pixel reading - PLAIN pydicom, unchanged from the v098 family's validated engine.
# ============================================================================================

def decode_rescaled(path):
    ds = pydicom.dcmread(str(path), force=True)
    img = ds.pixel_array.astype(np.float32, copy=False)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    img = img * slope + intercept
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        img = float(np.max(img)) - img
    return img, pixel_spacing(ds)


def pixel_spacing(ds):
    try:
        sp = ds.PixelSpacing
        return (float(sp[0]) + float(sp[1])) / 2.0
    except Exception:
        return np.nan


# ============================================================================================
# 5. Preprocessing - ordered slice sampling, physical crop, 2.5D triplet read_slot. Unchanged
# from the v098 family's validated engine.
# ============================================================================================

def slice_plan(n_z, d):
    n_z = int(n_z)
    d = int(d)
    if n_z <= 0:
        return np.zeros(d, np.int64), np.zeros(d, np.float32)
    if n_z >= d:
        idx = np.round(np.linspace(0, n_z - 1, d)).astype(np.int64)
        return np.clip(idx, 0, n_z - 1), np.ones(d, np.float32)
    pad = d - n_z
    left = pad // 2
    idx = np.concatenate([np.zeros(left, np.int64),
                          np.arange(n_z, dtype=np.int64),
                          np.full(pad - left, n_z - 1, np.int64)])
    m = np.concatenate([np.zeros(left, np.float32),
                        np.ones(n_z, np.float32),
                        np.zeros(pad - left, np.float32)])
    return idx, m


def pick_centres(n_z, d, rng):
    lo = int(math.floor(cfg.band_low * n_z))
    hi = int(math.ceil(cfg.band_high * n_z)) - 1
    lo = max(0, min(lo, n_z - 1))
    hi = max(0, min(hi, n_z - 1))
    if hi <= lo:
        return np.clip(np.arange(d) % max(n_z, 1), 0, n_z - 1)
    n_pos = hi - lo + 1
    slack = n_pos - d
    if slack <= 0:
        return np.clip(np.arange(lo, lo + d), lo, hi).astype(np.int64)
    if rng is None:
        idx = np.round(np.linspace(lo, hi, d)).astype(np.int64)
        return np.clip(idx, lo, hi)
    y = np.sort(rng.randint(0, slack + 1, size=d))
    return np.clip(lo + np.arange(d) + y, lo, hi).astype(np.int64)


def crop_window(h, w, spacing_mm):
    """physical_crop_resize's centre-crop as indices into the native slice - the CROP-FIRST
    split used by build_series_cache_v002_320_110.py (the v153 training shards).
    Returns (want, (y0, y1, x0, x1) clipped to the image, (pad_t, pad_b, pad_l, pad_r))."""
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        spacing_mm = cfg.crop_mm / max(h, 1)
    want = max(8, int(round(cfg.crop_mm / spacing_mm)))
    half = want // 2
    cy, cx = h // 2, w // 2
    y0, y1 = cy - half, cy - half + want
    x0, x1 = cx - half, cx - half + want
    pads = (max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w))
    return want, (max(0, y0), min(y1, h), max(0, x0), min(x1, w)), pads


def pad_resize(cropped, pads):
    pad_t, pad_b, pad_l, pad_r = pads
    out = cropped
    if pad_t or pad_b or pad_l or pad_r:
        out = np.pad(out, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant")
    interp = cv2.INTER_AREA if out.shape[0] > cfg.img_px else cv2.INTER_LINEAR
    return cv2.resize(out, (cfg.img_px, cfg.img_px), interpolation=interp)


def build_native_px(slices, spacing):
    """uint8 (n_z, img_px, img_px) for EVERY slice, in the order the v153 shard builder
    uses: CROP FIRST, then the 0.5/99.5 percentile window over the REAL cropped pixels
    (never the synthetic zero pad), then pad + resize. NOT v111's order (window over the
    full slice, then crop) - v153 trained on the shards, so this must match them."""
    n_z = len(slices)
    out = np.empty((n_z, cfg.img_px, cfg.img_px), dtype=np.uint8)
    cores, pads = [], []
    for s in slices:
        _, (y0, y1, x0, x1), pd_ = crop_window(s.shape[0], s.shape[1], spacing)
        cores.append(s[y0:y1, x0:x1].astype(np.float32))
        pads.append(pd_)
    if len({c.shape for c in cores}) == 1:
        lo, hi = np.percentile(np.stack(cores), [cfg.pct_low, cfg.pct_high])
    else:
        lo, hi = np.percentile(np.concatenate([c.ravel() for c in cores]),
                               [cfg.pct_low, cfg.pct_high])
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    for j in range(n_z):
        sl = np.clip((cores[j] - lo) / (hi - lo), 0.0, 1.0) * 255.0
        out[j] = pad_resize(sl, pads[j]).astype(np.uint8)
    return out


def read_slot(series_dir, flip_lr=False, reverse_z=False):
    """Mirrors train_v153's StudyDataset._from_cache in eval mode (rng=None):
    ALL native slices (no 48-slice cap), canonicalise (right knee: AXI/COR mirrored
    left-right, SAG stack reversed), crop-first window, then deterministic centres over the
    TRUE stack and +/- round(triplet_delta_mm / z_mm) native-index neighbours."""
    rec = scan_series(series_dir)
    ordered = rec["ordered"]
    if not ordered:
        raise FileNotFoundError(series_dir)
    spacing = rec["spacing"]
    z_mm = rec["z_mm"]
    if reverse_z:
        ordered = ordered[::-1]
    n_z = len(ordered)

    decoded, spac, bad = {}, [], []
    for k in range(n_z):
        try:
            img, sp = decode_rescaled(ordered[k])
        except Exception:
            bad.append(int(k))
            continue
        decoded[k] = img[:, ::-1] if flip_lr else img
        if np.isfinite(sp):
            spac.append(sp)
    if not decoded:
        raise RuntimeError(f"no decodable slices in {series_dir}")
    if bad:
        good = np.array(sorted(decoded), dtype=np.int64)
        for k in bad:
            decoded[k] = decoded[int(good[np.argmin(np.abs(good - k))])]

    if not np.isfinite(spacing):
        spacing = float(np.median(spac)) if spac else np.nan

    vol = build_native_px([decoded[k] for k in range(n_z)], spacing)
    del decoded

    d = cfg.cache_slices
    if n_z >= d:
        centres = pick_centres(n_z, d, None)
        smask = np.ones(d, np.float32)
    else:
        centres, smask = slice_plan(n_z, d)

    step = max(1, int(round(cfg.triplet_delta_mm / max(z_mm, 0.5))))
    idx = np.clip(centres[:, None] + np.array([-step, 0, step])[None, :], 0, n_z - 1)
    return vol[idx], smask


def preprocess_study(uid, series, image_root, laterality_workers=1):
    """Build the dense model input for one study without writing it to disk."""
    global _SER_SEL, IMG_ROOT, SLOT_LOOKUP, STUDY_LATERALITY, SERIES_SCAN
    IMG_ROOT = Path(image_root)
    _SER_SEL = series.loc[
        series.StudyInstanceUID.eq(uid)].copy()
    SERIES_SCAN = {}
    prescan(
        list(zip(
            _SER_SEL.StudyInstanceUID,
            _SER_SEL.SeriesInstanceUID)),
        workers=laterality_workers,
    )
    slots_df = build_slot_table()
    SLOT_LOOKUP = {
        study: {
            row.slot: (row.series_dir, int(row.n_slices))
            for row in group.itertuples()
        }
        for study, group in slots_df.groupby("StudyInstanceUID")
    }
    STUDY_LATERALITY = build_study_laterality()
    images = np.zeros((
        len(SLOTS), cfg.slices_per_slot, 3,
        cfg.img_px, cfg.img_px), dtype=np.uint8)
    mask = np.zeros(len(SLOTS), dtype=np.float32)
    smask = np.zeros(
        (len(SLOTS), cfg.slices_per_slot), dtype=np.float32)
    for slot_id, slot in enumerate(SLOTS):
        got = SLOT_LOOKUP.get(uid, {}).get(slot)
        if got is None:
            continue
        try:
            volume, real = read_slot(
                got[0], *flip_for(
                    uid, slot, STUDY_LATERALITY))
        except Exception as exc:
            print(
                f"Unusable series {uid}/{Path(got[0]).name}: "
                f"{type(exc).__name__}: {exc}", flush=True)
            continue
        images[slot_id] = volume
        smask[slot_id] = real
        mask[slot_id] = 1
    return images, mask, smask




# ============================================================================================
# 6. One study - what TestStudyDataset.__getitem__ builds, written to disk. Same loop, same
# read_slot / flip_for calls, with the resulting arrays written to disk.
# ============================================================================================

def process_study(uid):
    dst = OUT_DIR / f"{uid}.npz"
    if dst.exists():
        with np.load(dst, allow_pickle=False) as archive:
            if set(archive.files) != {
                    "imgs", "smask", "slot_names", "slot_ids",
                    "series_uid", "n_errors"}:
                raise ValueError(f"{dst}: unexpected archive keys")
            return (
                uid, "skipped", len(archive["slot_ids"]),
                int(archive["n_errors"]))
    imgs, smasks, names, ids, sers = [], [], [], [], []
    n_errors = 0
    for si, slot in enumerate(SLOTS):
        got = SLOT_LOOKUP.get(uid, {}).get(slot)
        if got is None:
            continue
        try:
            vol, real = read_slot(got[0], *flip_for(uid, slot, STUDY_LATERALITY))
        except Exception:
            n_errors += 1
            continue
        imgs.append(vol)
        smasks.append(real)
        names.append(slot)
        ids.append(si)
        sers.append(Path(got[0]).name)
    if not imgs:
        return uid, "no filled slots", 0, n_errors
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, imgs=np.stack(imgs), smask=np.stack(smasks),
             slot_names=np.array(names), slot_ids=np.array(ids, dtype=np.int64),
             series_uid=np.array(sers), n_errors=np.int64(n_errors))
    tmp.replace(dst)
    return uid, "ok", len(imgs), n_errors


# ============================================================================================
# 7. main() - the infer worker_main's data half: prescan -> slot table -> laterality, then the
# study loop.
# ============================================================================================

def main(argv=None):
    global _SER_SEL, SLOT_LOOKUP, STUDY_LATERALITY
    global INPUT_ROOT, OUT_DIR, WORKERS, STUDY_LIMIT, IMG_ROOT, SERIES_CSV

    parser = argparse.ArgumentParser(
        description="Build the processed RSNA training-study NPZ cache",
        allow_abbrev=False)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT,
                        help="directory containing train_series.csv and train_series/")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR,
                        help="destination for <StudyInstanceUID>.npz files")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="parallel study processes")
    parser.add_argument("--laterality-workers", type=int,
                        default=cfg.laterality_workers,
                        help="threads used for the DICOM header pass")
    parser.add_argument("--study-limit", type=int,
                        help="convert only the first N studies for a smoke test")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.laterality_workers < 1:
        parser.error("worker counts must be positive")
    if args.study_limit is not None and args.study_limit < 1:
        parser.error("--study-limit must be positive")

    INPUT_ROOT = args.input_root.resolve()
    OUT_DIR = args.output_dir.resolve()
    WORKERS = args.workers
    STUDY_LIMIT = args.study_limit
    cfg.laterality_workers = args.laterality_workers
    IMG_ROOT = INPUT_ROOT / "train_series"
    SERIES_CSV = INPUT_ROOT / "train_series.csv"
    missing = [path for path in (SERIES_CSV, IMG_ROOT) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Kaggle training data is incomplete under {INPUT_ROOT}: {missing}")

    ser_all = pd.read_csv(SERIES_CSV, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    studies = list(dict.fromkeys(ser_all.StudyInstanceUID))
    if STUDY_LIMIT is not None:
        studies = studies[:STUDY_LIMIT]
    _SER_SEL = ser_all[ser_all.StudyInstanceUID.isin(set(studies))].copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"input: {INPUT_ROOT}", flush=True)
    print(f"output: {OUT_DIR}", flush=True)
    print(f"header pass over {len(_SER_SEL)} series of {len(studies)} studies ...", flush=True)
    prescan(list(zip(_SER_SEL.StudyInstanceUID, _SER_SEL.SeriesInstanceUID)),
            workers=cfg.laterality_workers)
    slots_df = build_slot_table()
    SLOT_LOOKUP = {
        uid: {r.slot: (r.series_dir, int(r.n_slices)) for r in g.itertuples()}
        for uid, g in slots_df.groupby("StudyInstanceUID")
    }
    missing = sorted(set(studies) - set(slots_df.StudyInstanceUID))
    STUDY_LATERALITY = build_study_laterality()
    print(f"slot table: {len(slots_df)} series in slots, {len(missing)} studies missing; "
          f"laterality: {pd.Series(list(STUDY_LATERALITY.values())).fillna('none').value_counts().to_dict()}",
          flush=True)

    todo = [s for s in studies if s in SLOT_LOOKUP]
    counts, n_err, records = {}, 0, []
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for i, (uid, status, n_slots, ne) in enumerate(pool.imap_unordered(process_study, todo), 1):
            counts[status] = counts.get(status, 0) + 1
            n_err += ne
            records.append({
                "StudyInstanceUID": uid,
                "status": status,
                "n_slots": int(n_slots),
                "n_errors": int(ne),
            })
            if status not in ("ok", "skipped"):
                print(f"  {uid}: {status}", flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} {counts}", flush=True)
    manifest = {
        "schema_version": 1,
        "image_shape": [
            cfg.slices_per_slot, 3, cfg.img_px, cfg.img_px],
        "slots": SLOTS,
        "preprocessing": {
            "crop_mm": cfg.crop_mm,
            "pct_low": cfg.pct_low,
            "pct_high": cfg.pct_high,
            "triplet_delta_mm": cfg.triplet_delta_mm,
            "normalize_laterality": cfg.normalize_laterality,
        },
        "counts": counts,
        "slot_read_errors": n_err,
        "missing_from_slot_table": missing,
        "studies": sorted(
            records, key=lambda row: row["StudyInstanceUID"]),
    }
    temporary = OUT_DIR / "manifest.tmp.json"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUT_DIR / "manifest.json")
    print(f"done: {counts}  slot read errors: {n_err}  studies missing from slot table: {len(missing)}",
          flush=True)


if __name__ == "__main__":
    main()
