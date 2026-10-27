import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from skimage.transform import resize


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab.utils.prepare_knee import REPAIRS



def nearest_unique_indices(
    positions: list[float],
    targets: list[float],
) -> tuple[list[int], bool]:
    values = np.asarray(positions, dtype=np.float64)
    initial: list[int] = []
    for target in targets:
        right = int(np.searchsorted(values, target, side="left"))
        options = [
            index
            for index in (right - 1, right)
            if 0 <= index < len(positions)
        ]
        initial.append(
            min(
                options,
                key=lambda index: (
                    abs(positions[index] - target),
                    index,
                ),
            )
        )
    if len(set(initial)) == len(targets):
        return initial, False

    selected: list[int] = []
    previous = -1
    for target_index, target in enumerate(targets):
        last_allowed = len(positions) - (len(targets) - target_index)
        best = min(
            range(previous + 1, last_allowed + 1),
            key=lambda index: (
                abs(positions[index] - target),
                index,
            ),
        )
        selected.append(best)
        previous = best
    return selected, True


def select_series_slices(
    slices: list[dict[str, Any]], cfg,
) -> tuple[list[int], str, bool]:
    capacity = cfg.capacity_per_series
    threshold = cfg.long_series_threshold
    fraction = cfg.central_fraction
    count = len(slices)
    if count < 1:
        raise ValueError("Series contains no slices")
    if count <= capacity:
        return list(range(count)), "all_unique", False
    if count <= threshold:
        start = (count - capacity) // 2
        return (
            list(range(start, start + capacity)),
            "central_contiguous",
            False,
        )

    positions = [
        (
            float(item["spatial_position"])
            if item.get("spatial_position") is not None
            else None
        )
        for item in slices
    ]
    margin = (1 - fraction) / 2
    if (
        all(position is not None for position in positions)
        and float(positions[-1]) > float(positions[0])
    ):
        numeric = [float(position) for position in positions]
        span = numeric[-1] - numeric[0]
        targets = np.linspace(
            numeric[0] + margin * span,
            numeric[-1] - margin * span,
            capacity,
        ).tolist()
        selected, collision = nearest_unique_indices(
            numeric,
            targets,
        )
        return selected, "central_uniform_physical", collision

    targets = np.linspace(
        margin * (count - 1),
        (1 - margin) * (count - 1),
        capacity,
    ).tolist()
    selected, collision = nearest_unique_indices(
        list(range(count)),
        targets,
    )
    return selected, "central_uniform_index_fallback", collision


def resize_uint16(array: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999999:
        return array
    height = max(1, round(array.shape[0] * scale))
    width = max(1, round(array.shape[1] * scale))
    result = resize(
        array.astype(np.float32),
        (height, width),
        order=3,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.rint(np.clip(result, 0, 65535)).astype(np.uint16)


def encode_png(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    image = Image.fromarray(array)
    try:
        if image.mode != "I;16":
            raise ValueError(f"Expected I;16 image, received {image.mode}")
        image.save(output, format="PNG", compress_level=9)
    finally:
        image.close()
    return output.getvalue()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fit_request(records, arrays, cfg, prompt):
    encoded = [encode_png(array) for array in arrays]
    original_bytes = prompt.serialized_request_bytes(records, encoded, **cfg.generation)
    if original_bytes <= cfg.target_request_body_bytes:
        return 1.0, encoded, original_bytes, original_bytes
    base64_bytes = sum(4 * math.ceil(len(image) / 3) for image in encoded)
    available = max(1, cfg.target_request_body_bytes - (original_bytes - base64_bytes))
    initial = min(0.999, max(0.05, math.sqrt(available / base64_bytes) * 0.99))

    def render(scale):
        images = [encode_png(resize_uint16(array, scale)) for array in arrays]
        return images, prompt.serialized_request_bytes(records, images, **cfg.generation)

    images, size = render(initial)
    if size <= cfg.target_request_body_bytes:
        lower, upper = initial, 1.0
        best = initial, images, original_bytes, size
    else:
        lower, upper = 0.05, initial
        best = None
    for _ in range(6):
        scale = (lower + upper) / 2
        images, size = render(scale)
        if size <= cfg.target_request_body_bytes:
            best = scale, images, original_bytes, size
            lower = scale
        else:
            upper = scale
    if best is None:
        images, size = render(lower)
        if size > cfg.target_request_body_bytes:
            raise ValueError("Study cannot fit the request-body budget")
        best = lower, images, original_bytes, size
    return best


def prepare_study(uid, series, metadata, cfg, prompt, identity, reuse_identities=()):
    directory = ROOT / cfg.image_dir / uid
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["preparation_identity"] != identity and manifest["preparation_identity"] not in reuse_identities:
            raise ValueError(f"Prepared image recipe changed: {uid}")
        return manifest
    records, arrays, summaries = [], [], []
    tags = ["Rows", "Columns", "SeriesDescription", "SOPInstanceUID", "InstanceNumber",
            "ImageOrientationPatient", "ImagePositionPatient"]
    for row in series.sort_values("SeriesInstanceUID").itertuples(index=False):
        series_uid = row.SeriesInstanceUID
        source = ROOT / cfg.npz_dir / uid / f"{series_uid}.npz"
        meta = metadata.loc[series_uid]
        paths = sorted((ROOT / cfg.dicom_dir / uid / series_uid).glob("*.dcm"))
        headers = [(path, pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags))
                   for path in paths if path.stem != REPAIRS.get(series_uid)]
        if meta.order_method == "image_position_patient_normal":
            def spatial_position(item):
                header = item[1]
                normal = np.cross(header.ImageOrientationPatient[:3], header.ImageOrientationPatient[3:])
                return float(np.dot(header.ImagePositionPatient, normal))
            headers.sort(key=lambda item: (spatial_position(item), float(item[1].InstanceNumber), str(item[0])))
            positions = [spatial_position(item) for item in headers]
        elif meta.order_method == "instance_number":
            headers.sort(key=lambda item: (float(item[1].InstanceNumber), str(item[0])))
            positions = [None] * len(headers)
        elif meta.order_method == "filename_fallback":
            positions = [None] * len(headers)
        else:
            raise ValueError(f"Unsupported cache order: {meta.order_method}")
        with np.load(source) as archive:
            volume = archive["volume"]
        expected = (len(headers), int(headers[0][1].Rows), int(headers[0][1].Columns))
        if volume.dtype != np.uint16 or volume.shape != expected:
            raise ValueError(f"Cache/DICOM dimensions differ: {source}")
        entries = [{"spatial_index": index + 1, "spatial_position": position}
                   for index, position in enumerate(positions)]
        indices, policy, collision = select_series_slices(entries, cfg)
        summaries.append({
            "series_instance_uid": series_uid, "source_npz": str(source.relative_to(ROOT)),
            "source_npz_sha256": sha256(source), "source_shape": list(volume.shape),
            "value_scale": float(meta.value_scale), "value_offset": float(meta.value_offset),
            "order_method": meta.order_method, "selection_policy": policy,
            "selection_collision_resolved": collision,
            "selected_spatial_indices": [index + 1 for index in indices],
        })
        for index in indices:
            records.append({
                "series": series_uid, "series_description": str(getattr(headers[index][1], "SeriesDescription", "")),
                "plane": row.Anatomical_Plane, "spatial_index": index + 1,
                "spatial_position": positions[index], "series_slice_count": len(headers),
                "sop_instance_uid": str(headers[index][1].SOPInstanceUID),
                "image_relative_path": f"{series_uid}/{index + 1:04d}.png",
            })
            arrays.append(volume[index].copy())
        del volume
    scale, images, original_bytes, final_bytes = fit_request(records, arrays, cfg, prompt)
    for record, array, image in zip(records, arrays, images, strict=True):
        path = directory / record["image_relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image)
        record.update(
            image_sha256=hashlib.sha256(image).hexdigest(), bytes=len(image),
            width=max(1, round(array.shape[1] * scale)), height=max(1, round(array.shape[0] * scale)),
            mode="I;16", bit_depth=16,
        )
    manifest = {
        "StudyInstanceUID": uid, "preparation_identity": identity,
        "pixel_recipe": "existing_per_series_minmax_uint16_cache_native_png",
        "original_request_body_bytes": original_bytes, "final_request_body_bytes": final_bytes,
        "study_scale": scale, "series": summaries, "images": records,
    }
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', default='assets/data/rsna-knee-abnormality-detection/train.csv')
    parser.add_argument('--output-dir', default='outputs/labeling/individual_images')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    from types import SimpleNamespace
    from labeling.prompts import prompt_048 as prompt
    cfg = SimpleNamespace(input_csv=args.input_csv, image_dir=args.output_dir,
                          series_csv='assets/data/rsna-knee-abnormality-detection/train_series.csv',
                          dicom_dir='assets/data/rsna-knee-abnormality-detection/train_series',
                          npz_dir='assets/data/parag/train_series_npz_uint16',
                          npz_metadata=('assets/data/parag/train_series_npz_uint16_metadata.csv',),
                          capacity_per_series=9, long_series_threshold=18, central_fraction=.8,
                          target_request_body_bytes=25500000, hard_request_body_bytes=26000000,
                          preparation_workers=args.workers, generation={'max_tokens': 8192, 'reasoning_effort': 'medium'})
    studies = pd.read_csv(ROOT / cfg.input_csv).StudyInstanceUID.tolist()
    series = pd.read_csv(ROOT / cfg.series_csv)
    metadata = pd.concat([pd.read_csv(ROOT / path) for path in cfg.npz_metadata])
    metadata = metadata.drop_duplicates("SeriesInstanceUID", keep="last").set_index("SeriesInstanceUID")
    identity = {
        "code_sha256": sha256(Path(__file__)), "prompt_sha256": sha256(Path(prompt.__file__)),
        "source_code_sha256": sha256(ROOT / "lab/utils/prepare_knee.py"),
        "input_sha256": sha256(ROOT / cfg.input_csv), "series_sha256": sha256(ROOT / cfg.series_csv),
        "npz_metadata_sha256": {path: sha256(ROOT / path) for path in cfg.npz_metadata},
        "recipe": {key: getattr(cfg, key) for key in (
            "capacity_per_series", "long_series_threshold", "central_fraction",
            "target_request_body_bytes", "hard_request_body_bytes", "generation",
        )},
    }
    reuse_identities = []
    for name, expected_hash in getattr(cfg, "preparation_cache_manifests", {}).items():
        path = ROOT / name
        if sha256(path) != expected_hash:
            raise ValueError(f"Prepared input manifest hash mismatch: {path}")
        cached_identity = json.loads(path.read_text())["identity"]
        for key in ("series_sha256", "npz_metadata_sha256", "source_code_sha256", "recipe"):
            if cached_identity[key] != identity[key]:
                raise ValueError(f"Prepared input recipe differs: {key}")
        reuse_identities.append(cached_identity)
    identity["reused_preparation_manifests"] = getattr(cfg, "preparation_cache_manifests", {})
    results = []
    with ThreadPoolExecutor(max_workers=cfg.preparation_workers) as pool:
        futures = [pool.submit(prepare_study, uid, series.loc[series.StudyInstanceUID == uid],
                               metadata, cfg, prompt, identity, reuse_identities) for uid in studies]
        for future in as_completed(futures):
            manifest = future.result()
            results.append(manifest)
            print(f"{len(results)}/{len(studies)} prepared; images={len(manifest['images'])} "
                  f"scale={manifest['study_scale']:.4f}", flush=True)
    output = ROOT / cfg.image_dir
    destination = ROOT / getattr(cfg, "preparation_manifest", str(output / "manifest.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, {
        "identity": identity, "studies": len(results),
        "series": sum(len(result["series"]) for result in results),
        "selected_images": sum(len(result["images"]) for result in results),
        "resized_studies": sum(result["study_scale"] < 0.999999 for result in results),
        "maximum_request_body_bytes": max(result["final_request_body_bytes"] for result in results),
    })
    print(output, flush=True)


if __name__ == "__main__":
    main()
