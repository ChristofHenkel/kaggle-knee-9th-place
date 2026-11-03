#!/usr/bin/env python3
"""Convert RSNA knee DICOM series to ordered 8-bit PNG series.

The script performs a complete metadata-only preflight before writing pixels.
It refuses conversion when a listed file would be dropped, native shapes vary
within a series, or ImageOrientationPatient disagrees with the supplied plane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from skp.toolbox.dicom import (
    anatomical_plane_from_iop,
    inspect_dicom_series,
    load_dicom_series,
)


ALGORITHM_VERSION = "series-p01-p99-uint8-geometry-rank-v1"
DEFAULT_DATA_ROOT = Path(
    "assets/data/rsna-knee-abnormality-detection"
)
DEFAULT_OUTPUT_ROOT = Path(
    "assets/data/maxvit_png"
)


@dataclass(frozen=True)
class SeriesSpec:
    split: str
    study_uid: str
    series_uid: str
    expected_plane: str
    input_dir: str
    output_dir: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "test"), default=["train", "test"]
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--png-compression", type=int, default=1, choices=range(10))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--skip-malformed-series",
        action="store_true",
        help="Continue after pixel-decode/conversion failures and record every failure.",
    )
    return parser.parse_args()


def series_specs(
    data_root: Path, output_root: Path, splits: list[str]
) -> tuple[list[SeriesSpec], dict[str, str]]:
    specs = []
    hashes = {}
    for split in splits:
        csv_path = data_root / f"{split}_series.csv"
        frame = pd.read_csv(csv_path)
        required = {
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Anatomical_Plane",
        }
        if not required <= set(frame.columns):
            raise ValueError(f"{csv_path} is missing {sorted(required - set(frame.columns))}")
        if not frame["SeriesInstanceUID"].is_unique:
            raise ValueError(f"SeriesInstanceUID is not unique in {csv_path}")
        hashes[csv_path.name] = sha256_file(csv_path)
        for row in frame.itertuples(index=False):
            specs.append(
                SeriesSpec(
                    split=split,
                    study_uid=row.StudyInstanceUID,
                    series_uid=row.SeriesInstanceUID,
                    expected_plane=row.Anatomical_Plane,
                    input_dir=str(
                        data_root
                        / f"{split}_series"
                        / row.StudyInstanceUID
                        / row.SeriesInstanceUID
                    ),
                    output_dir=str(
                        output_root
                        / f"{split}_series"
                        / row.StudyInstanceUID
                        / row.SeriesInstanceUID
                    ),
                )
            )
    return specs, hashes


def _metadata_plane(metadata: dict[str, Any]) -> str:
    return anatomical_plane_from_iop(metadata["ImageOrientationPatient"])


def preflight_one(spec: SeriesSpec) -> dict[str, Any]:
    try:
        inspected = inspect_dicom_series(spec.input_dir, max_workers=1)
        metadata = inspected["slice_metadata"]
        planes = {_metadata_plane(item) for item in metadata}
        shapes = {(int(item["Rows"]), int(item["Columns"])) for item in metadata}
        errors = []
        if inspected["invalid_files"]:
            errors.append(f"{len(inspected['invalid_files'])} files would be dropped")
        if len(inspected["input_files"]) != len(metadata):
            errors.append(
                f"{len(inspected['input_files'])} input files != {len(metadata)} valid slices"
            )
        if planes != {spec.expected_plane}:
            errors.append(
                f"ImageOrientationPatient planes {sorted(planes)} != "
                f"label {spec.expected_plane}"
            )
        if len(shapes) != 1:
            errors.append(f"native shapes vary within series: {sorted(shapes)}")
        positions = [float(item["z_location"]) for item in metadata]
        if any(right <= left for left, right in zip(positions, positions[1:])):
            errors.append("geometry-derived slice positions are not strictly increasing")
        return {
            "ok": not errors,
            "split": spec.split,
            "study_uid": spec.study_uid,
            "series_uid": spec.series_uid,
            "expected_plane": spec.expected_plane,
            "slices": len(metadata),
            "errors": errors,
        }
    except Exception as error:
        return {
            "ok": False,
            "split": spec.split,
            "study_uid": spec.study_uid,
            "series_uid": spec.series_uid,
            "expected_plane": spec.expected_plane,
            "slices": 0,
            "errors": [f"{type(error).__name__}: {error}"],
        }


def run_preflight(
    specs: list[SeriesSpec], workers: int, output_root: Path, input_hashes: dict[str, str]
) -> dict[str, Any]:
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(preflight_one, specs), 1):
            results.append(result)
            if index % 500 == 0 or index == len(specs):
                failures = sum(not item["ok"] for item in results)
                print(
                    f"Preflight {index:,}/{len(specs):,} series; "
                    f"failures={failures:,}",
                    flush=True,
                )
    failures = [result for result in results if not result["ok"]]
    plane_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for result in results:
        plane_counts[result["expected_plane"]] = (
            plane_counts.get(result["expected_plane"], 0) + 1
        )
        split_counts[result["split"]] = split_counts.get(result["split"], 0) + 1
    report = {
        "algorithm_version": ALGORITHM_VERSION,
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "input_csv_sha256": input_hashes,
        "series": len(results),
        "slices": sum(result["slices"] for result in results),
        "failures": len(failures),
        "series_by_split": split_counts,
        "series_by_plane": plane_counts,
        "failure_details": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def percentile_clip_uint8(
    volume: np.ndarray, lower_percentile: float = 1, upper_percentile: float = 99
) -> tuple[np.ndarray, float, float]:
    """Clip a volume by series-wide percentiles and linearly map it to uint8."""
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
    lower, upper = np.percentile(volume, [lower_percentile, upper_percentile])
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite([lower, upper]).all():
        raise ValueError("Non-finite percentile threshold encountered.")
    if upper <= lower:
        return np.zeros(volume.shape, dtype=np.uint8), lower, upper
    clipped = np.clip(volume, lower, upper)
    scaled = np.rint((clipped - lower) * (255.0 / (upper - lower)))
    return scaled.astype(np.uint8), lower, upper


def ordered_png_name(rank: int, source_path: str, width: int) -> str:
    return f"{rank:0{width}d}__{Path(source_path).stem}.png"


def _completed_series(output_dir: Path) -> dict[str, Any] | None:
    marker = output_dir / "conversion.json"
    if not output_dir.exists():
        return None
    if not marker.is_file():
        raise FileExistsError(f"Output exists without completion marker: {output_dir}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    png_count = sum(path.suffix.lower() == ".png" for path in output_dir.iterdir())
    if metadata.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError(f"Output algorithm version differs: {output_dir}")
    if png_count != metadata.get("slices"):
        raise ValueError(
            f"Output PNG count {png_count} != marker {metadata.get('slices')}: "
            f"{output_dir}"
        )
    return metadata


def convert_one(task: tuple[SeriesSpec, int]) -> dict[str, Any]:
    spec, png_compression = task
    output_dir = Path(spec.output_dir)
    try:
        completed = _completed_series(output_dir)
        if completed is not None:
            return {
                "ok": True,
                "skipped": True,
                "split": spec.split,
                "series_uid": spec.series_uid,
                "slices": completed["slices"],
                "bytes": completed["png_bytes"],
            }

        loaded = load_dicom_series(spec.input_dir, max_workers=1)
        metadata = loaded["slice_metadata"]
        planes = {_metadata_plane(item) for item in metadata}
        if loaded["invalid_files"] or len(metadata) != len(loaded["sorted_files"]):
            raise ValueError("DICOM loader dropped or miscounted input files")
        if planes != {spec.expected_plane}:
            raise ValueError(
                f"ImageOrientationPatient planes {sorted(planes)} != "
                f"label {spec.expected_plane}"
            )

        images, p01, p99 = percentile_clip_uint8(loaded["image"], 1, 99)
        width = max(4, len(str(len(images) - 1)))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{spec.series_uid}.tmp-", dir=output_dir.parent)
        )
        try:
            png_names = []
            png_bytes = 0
            for rank, (image, source) in enumerate(
                zip(images, loaded["sorted_files"], strict=True)
            ):
                name = ordered_png_name(rank, source, width)
                destination = temporary / name
                written = cv2.imwrite(
                    str(destination),
                    image,
                    [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
                )
                if not written:
                    raise IOError(f"OpenCV failed to write {destination}")
                png_names.append(name)
                png_bytes += destination.stat().st_size
            if png_names != sorted(png_names):
                raise ValueError("PNG filenames do not preserve sorted slice order")
            marker = {
                "algorithm_version": ALGORITHM_VERSION,
                "completed_at": utc_now(),
                "split": spec.split,
                "study_uid": spec.study_uid,
                "series_uid": spec.series_uid,
                "anatomical_plane": spec.expected_plane,
                "sort": "ascending ImagePositionPatient projected onto IOP normal",
                "percentiles": [1, 99],
                "p01": p01,
                "p99": p99,
                "slices": len(images),
                "png_bytes": png_bytes,
            }
            (temporary / "conversion.json").write_text(
                json.dumps(marker, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, output_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {
            "ok": True,
            "skipped": False,
            "split": spec.split,
            "series_uid": spec.series_uid,
            "slices": len(images),
            "bytes": png_bytes,
        }
    except Exception as error:
        return {
            "ok": False,
            "skipped": False,
            "split": spec.split,
            "study_uid": spec.study_uid,
            "series_uid": spec.series_uid,
            "input_dir": spec.input_dir,
            "slices": 0,
            "bytes": 0,
            "error": f"{type(error).__name__}: {error}",
        }


def run_conversion(
    specs: list[SeriesSpec],
    workers: int,
    png_compression: int,
    output_root: Path,
    skip_malformed_series: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    progress_path = output_root / "conversion_progress.jsonl"
    processed_series = 0
    successful_series = 0
    total_slices = 0
    total_bytes = 0
    skipped_series = 0
    failures = []
    stopped_early = False
    iterator = iter(specs)
    pending = {}

    with progress_path.open("a", encoding="utf-8") as progress, ProcessPoolExecutor(
        max_workers=workers
    ) as pool:
        for _ in range(min(workers * 2, len(specs))):
            spec = next(iterator, None)
            if spec is None:
                break
            pending[pool.submit(convert_one, (spec, png_compression))] = spec

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                progress.write(json.dumps(result, separators=(",", ":")) + "\n")
                progress.flush()
                processed_series += 1
                if not result["ok"]:
                    failures.append(result)
                    if not skip_malformed_series:
                        stopped_early = True
                        for queued in pending:
                            queued.cancel()
                        pending.clear()
                        break
                else:
                    successful_series += 1
                    total_slices += int(result["slices"])
                    total_bytes += int(result["bytes"])
                    skipped_series += int(result["skipped"])
                if processed_series % 100 == 0 or processed_series == len(specs):
                    elapsed = time.monotonic() - started
                    rate = total_slices / elapsed if elapsed else 0
                    print(
                        f"Processed {processed_series:,}/{len(specs):,} series, "
                        f"{total_slices:,} slices ({rate:.1f} slices/s), "
                        f"existing={skipped_series:,}, failures={len(failures):,}",
                        flush=True,
                    )
                spec = next(iterator, None)
                if spec is not None:
                    pending[pool.submit(convert_one, (spec, png_compression))] = spec
            if stopped_early:
                break

    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "requested_series": len(specs),
        "processed_series": processed_series,
        "successful_series": successful_series,
        "skipped_existing_series": skipped_series,
        "failed_series": len(failures),
        "png_files": total_slices,
        "png_bytes": total_bytes,
        "skip_malformed_series": skip_malformed_series,
        "stopped_early": stopped_early,
        "failure_details": failures,
    }
    (output_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "failed_series.json").write_text(
        json.dumps(
            {
                "algorithm_version": ALGORITHM_VERSION,
                "completed_at": summary["completed_at"],
                "failed_series": len(failures),
                "failure_details": failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if data_root == output_root:
        raise ValueError("Input and output roots must differ")
    specs, input_hashes = series_specs(data_root, output_root, args.splits)
    print(
        f"Preflighting {len(specs):,} series before any PNG conversion...",
        flush=True,
    )
    preflight = run_preflight(specs, args.workers, output_root, input_hashes)
    if preflight["failures"]:
        print(
            f"Preflight failed for {preflight['failures']:,} series; no PNG conversion started.",
            flush=True,
        )
        return 2
    print(
        f"Preflight passed: {preflight['series']:,} series, "
        f"{preflight['slices']:,} slices, all orientations match labels.",
        flush=True,
    )
    if args.preflight_only:
        return 0
    summary = run_conversion(
        specs,
        args.workers,
        args.png_compression,
        output_root,
        skip_malformed_series=args.skip_malformed_series,
    )
    if summary["stopped_early"]:
        print(
            f"Conversion stopped after an error: {summary['failure_details'][0]}",
            flush=True,
        )
        return 3
    print(
        f"Conversion complete: {summary['png_files']:,} PNGs in "
        f"{summary['elapsed_seconds']:.1f}s; "
        f"failed series={summary['failed_series']:,}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
