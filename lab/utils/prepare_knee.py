from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPAIRS = {
    "1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348": "1.2.826.0.1.3680043.8.498.41947538458330624797291612781685746320",
    "1.2.826.0.1.3680043.8.498.31160785238781353848727311763596115703": "1.2.826.0.1.3680043.8.498.66460523372602171398246641420761543626",
}

def natural_path_key(path: Path):
    return str(path)

def header_record(path: Path) -> dict[str, Any]:
    try:
        ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        if iop.size < 6 or ipp.size < 3:
            raise ValueError('missing ImageOrientationPatient/ImagePositionPatient')
        normal = np.cross(iop[:3], iop[3:6])
        k = float(np.dot(ipp[:3], normal))
        instance = float(getattr(ds, 'InstanceNumber', np.inf))
        return {'path': path, 'header_ok': 1, 'k': k, 'instance_number': instance, 'ipp_x': float(ipp[0]), 'ipp_y': float(ipp[1]), 'ipp_z': float(ipp[2]), 'iop': [float(x) for x in iop[:6]]}
    except Exception as exc:
        return {'path': path, 'header_ok': 0, 'k': np.nan, 'instance_number': np.inf, 'error': repr(exc)}

def order_paths(paths: list[Path]) -> tuple[list[Path], str, int]:
    headers = [header_record(path) for path in paths]
    need = max(2, int(0.8 * len(headers)))
    geometric = [h for h in headers if h['header_ok'] and np.isfinite(h['k'])]
    if len(geometric) >= need:
        ordered = sorted(headers, key=lambda h: (h['k'] if np.isfinite(h['k']) else np.inf, h['instance_number'], natural_path_key(h['path'])))
        return ([h['path'] for h in ordered], 'image_position_patient_normal', 1)
    numbered = [h for h in headers if np.isfinite(h['instance_number'])]
    if len(numbered) >= need:
        ordered = sorted(headers, key=lambda h: (h['instance_number'], natural_path_key(h['path'])))
        return ([h['path'] for h in ordered], 'instance_number', 0)
    return (sorted(paths, key=natural_path_key), 'filename_fallback', 0)

def decode_rescaled(path: Path, apply_monochrome1: bool) -> tuple[np.ndarray, dict[str, Any]]:
    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array.astype(np.float32, copy=False)
    slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
    image = image * slope + intercept
    photometric = str(getattr(ds, 'PhotometricInterpretation', '')).upper()
    if apply_monochrome1 and photometric == 'MONOCHROME1':
        image = float(np.max(image)) - image
    spacing = getattr(ds, 'PixelSpacing', [np.nan, np.nan])
    return (image, {'Rows': int(image.shape[0]), 'Columns': int(image.shape[1]), 'PixelSpacing_row': float(spacing[0]) if len(spacing) > 0 else np.nan, 'PixelSpacing_col': float(spacing[1]) if len(spacing) > 1 else np.nan, 'SliceThickness': float(getattr(ds, 'SliceThickness', np.nan)), 'PhotometricInterpretation': photometric, 'TransferSyntaxUID': str(getattr(getattr(ds, 'file_meta', None), 'TransferSyntaxUID', ''))})

def encode_uint16(volume: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(volume)
    if not finite.all():
        raise ValueError('decoded volume contains non-finite values')
    value_min = float(volume.min())
    value_max = float(volume.max())
    if value_max <= value_min:
        return (np.zeros(volume.shape, dtype=np.uint16), 1.0, value_min)
    scale = (value_max - value_min) / 65535.0
    encoded = np.rint((volume - value_min) / scale).clip(0, 65535).astype(np.uint16)
    return (encoded, float(scale), value_min)

def convert_series(task):
    study, series, source_root, destination = task
    out = Path(destination) / "train_series_npz_uint16" / study / f"{series}.npz"
    record = out.with_suffix(".json")
    if record.exists() and out.exists():
        return json.loads(record.read_text())
    directory = Path(source_root) / study / series
    paths = sorted(directory.glob("*.dcm"))
    ordered, method, geometry_ok = order_paths(paths)
    bad = REPAIRS.get(series)
    kept = [p for p in ordered if p.stem != bad]
    arrays, first_meta = [], None
    for path in kept:
        image, meta = decode_rescaled(path, True)
        if arrays and image.shape != arrays[0].shape:
            raise ValueError(f"inconsistent slice shape: {directory}: {image.shape}")
        if first_meta is None:
            first_meta = meta
        arrays.append(image)
    volume = np.stack(arrays)
    encoded, scale, offset = encode_uint16(volume)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, volume=encoded)
    os.replace(temporary, out)
    result = {
        "StudyInstanceUID": study, "SeriesInstanceUID": series,
        "output_path": str(out), "status": "converted",
        "output_n_slices": len(encoded), "n_dicom": len(paths),
        "skipped_n_slices": len(ordered) - len(kept),
        "order_method": method, "spatial_order_ok": geometry_ok,
        "value_scale": scale, "value_offset": offset,
        **first_meta,
    }
    temporary_record = record.with_suffix(".tmp.json")
    temporary_record.write_text(json.dumps(result) + "\n")
    os.replace(temporary_record, record)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    source = PROJECT_ROOT / "assets/data/rsna-knee-abnormality-detection"
    destination = PROJECT_ROOT / "assets/data/parag"
    destination.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(source / "train_series.csv")
    if args.limit:
        series = series.iloc[:args.limit]
    tasks = [(r.StudyInstanceUID, r.SeriesInstanceUID, str(source / "train_series"), str(destination))
             for r in series.itertuples()]
    rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_series, task): task[:2] for task in tasks}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 100 == 0 or len(rows) == len(tasks):
                print(f"converted {len(rows)}/{len(tasks)} elapsed={time.monotonic() - started:.1f}s", flush=True)
    converted = pd.DataFrame(rows)
    converted.sort_values(['StudyInstanceUID', 'SeriesInstanceUID']).to_csv(
        destination / 'train_series_npz_uint16_metadata.csv', index=False)
    print("conversion complete", flush=True)


if __name__ == "__main__":
    main()
