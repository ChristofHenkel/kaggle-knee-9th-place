"""DICOM loading helpers with lazy optional dependencies."""

from __future__ import annotations

import concurrent.futures
import os
from collections import Counter
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch


def _require_pydicom():
    try:
        import pydicom
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "pydicom is required for DICOM loading. Install it with `uv add pydicom`."
        ) from e
    return pydicom


def _get_dicomsdl():
    try:
        import dicomsdl
    except ModuleNotFoundError:
        return None
    return dicomsdl


def apply_windowing(
    image: np.ndarray | torch.Tensor,
    windows: list[tuple[float, float]]
    | tuple[float, float]
    | np.ndarray
    | torch.Tensor,
    to_uint8: bool = True,
    device: str | torch.device | None = None,
) -> np.ndarray | torch.Tensor:
    """Apply CT windowing to a NumPy array or torch tensor.

    Args:
        image: Input HU array/tensor with arbitrary shape.
        windows: One ``(window_level, window_width)`` pair or many pairs.
        to_uint8: If True, scale the normalized output to 0-255 and cast uint8.
        device: Optional torch device. NumPy inputs only support CPU.
    """
    is_numpy = isinstance(image, np.ndarray)
    if isinstance(windows, tuple):
        windows = [windows]
    if isinstance(windows, list):
        windows = np.asarray(windows, dtype=np.float32)

    if is_numpy:
        if device is not None and str(device) != "cpu":
            raise ValueError("NumPy windowing only supports CPU execution.")
        image = image.astype(np.float32, copy=False)
        windows = windows.astype(np.float32, copy=False)
        if windows.ndim == 1:
            windows = windows[None, :]
        image_expanded = image[..., np.newaxis]
        shape = [1] * image.ndim + [len(windows)]
        wl = windows[..., 0].reshape(shape)
        ww = windows[..., 1].reshape(shape)
        out = np.clip(image_expanded, wl - ww / 2, wl + ww / 2)
        out = (out - (wl - ww / 2)) / (ww + 1e-6)
        if to_uint8:
            out = (out * 255.0).astype(np.uint8)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    if device is not None:
        image = image.to(device=device)
    if isinstance(windows, np.ndarray):
        windows = torch.from_numpy(windows).to(device=image.device, dtype=image.dtype)
    elif isinstance(windows, torch.Tensor):
        windows = windows.to(device=image.device, dtype=image.dtype)
    image = image.float()
    if windows.ndim == 1:
        windows = windows.unsqueeze(0)
    image_expanded = image.unsqueeze(-1)
    shape = [1] * image.ndim + [len(windows)]
    wl = windows[..., 0].view(shape)
    ww = windows[..., 1].view(shape)
    out = torch.clamp(image_expanded, min=wl - ww / 2, max=wl + ww / 2)
    out.sub_(wl - ww / 2).div_(ww + 1e-6)
    if to_uint8:
        out = out.mul(255.0).to(torch.uint8)
    return out.squeeze(-1) if out.shape[-1] == 1 else out


def load_dicom_series(
    folder_path: str,
    backend: str = "pydicom",
    sort_by_instance: bool = False,
    rescale_pixel_values: bool = True,
    fix_unequal_shapes_method: str = "crop_pad",
    max_workers: int | None = None,
    orientation: str | None = None,
) -> Dict:
    """Load a single-frame DICOM series into a volume.

    The function filters unreadable files, localizer/scout images, mismatched
    series UIDs, duplicate slice positions, and slices with inconsistent
    anatomical planes. It returns the volume plus metadata needed for debugging
    sorting and orientation issues.
    """
    if backend not in {"pydicom", "dicomsdl"}:
        raise ValueError("backend must be 'pydicom' or 'dicomsdl'.")
    inspection = inspect_dicom_series(
        folder_path,
        backend=backend,
        sort_by_instance=sort_by_instance,
        max_workers=max_workers,
    )
    backend = inspection["backend"]
    valid_slices = inspection["slice_metadata"]
    invalid_files = inspection["invalid_files"]
    pydicom = _require_pydicom()
    dicomsdl = _get_dicomsdl()
    if max_workers is None:
        max_workers = os.cpu_count()

    target_shape = Counter((m["Rows"], m["Columns"]) for m in valid_slices).most_common(1)[0][0]
    worker_args = [
        (meta, backend, pydicom, dicomsdl, rescale_pixel_values, target_shape, fix_unequal_shapes_method)
        for meta in valid_slices
    ]

    volume = np.zeros((len(valid_slices), *target_shape), dtype=np.float32)
    if max_workers == 1:
        arrays = map(_read_pixel_worker, worker_args)
        for idx, arr in enumerate(arrays):
            if arr is None:
                raise IOError(f"Failed to read pixel data: {valid_slices[idx]['path']}")
            volume[idx] = arr
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, arr in enumerate(executor.map(_read_pixel_worker, worker_args)):
                if arr is None:
                    raise IOError(
                        f"Failed to read pixel data: {valid_slices[idx]['path']}"
                    )
                volume[idx] = arr

    spacing, orientation_code, all_ipps = _series_geometry(valid_slices)
    final_code = orientation_code
    if orientation is not None:
        volume, spacing = _reorient_array(volume, spacing, orientation_code, orientation)
        final_code = orientation

    return {
        "image": volume,
        "spacing": spacing,
        "orientation_code": final_code,
        "modality": valid_slices[0]["Modality"],
        "image_position_patient": all_ipps,
        "sorted_files": [m["path"] for m in valid_slices],
        "slice_metadata": valid_slices,
        "invalid_files": invalid_files,
    }


def inspect_dicom_series(
    folder_path: str,
    backend: str = "pydicom",
    sort_by_instance: bool = False,
    max_workers: int | None = None,
) -> Dict:
    """Read, validate, and geometrically order a series without decoding pixels."""
    if backend not in {"pydicom", "dicomsdl"}:
        raise ValueError("backend must be 'pydicom' or 'dicomsdl'.")
    pydicom = _require_pydicom()
    dicomsdl = _get_dicomsdl()
    if backend == "dicomsdl" and dicomsdl is None:
        print("dicomsdl not found; falling back to pydicom.")
        backend = "pydicom"
    if max_workers is None:
        max_workers = os.cpu_count()

    files = [str(path) for path in Path(folder_path).rglob("*") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in {folder_path}")

    if max_workers == 1:
        metadata = [
            _read_metadata_worker(path, backend, pydicom, dicomsdl) for path in files
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            metadata = list(
                executor.map(
                    lambda path: _read_metadata_worker(
                        path, backend, pydicom, dicomsdl
                    ),
                    files,
                )
            )

    metadata_list = [meta for meta in metadata if meta is not None]
    invalid_files = [path for path, meta in zip(files, metadata) if meta is None]
    if not metadata_list:
        raise ValueError(f"No valid DICOM headers could be read in {folder_path}.")

    valid_slices, invalid_files = _filter_valid_slices(metadata_list, invalid_files)
    valid_slices = _sort_and_dedupe_slices(
        valid_slices, invalid_files, sort_by_instance
    )
    return {
        "backend": backend,
        "input_files": files,
        "slice_metadata": valid_slices,
        "sorted_files": [meta["path"] for meta in valid_slices],
        "invalid_files": invalid_files,
    }


def anatomical_plane_from_iop(image_orientation_patient) -> str:
    """Map DICOM row/column direction cosines to the nearest anatomical plane."""
    iop = np.asarray(image_orientation_patient, dtype=float)
    if iop.shape != (6,):
        raise ValueError(f"ImageOrientationPatient must have 6 values, got {iop.shape}.")
    normal = np.cross(iop[:3], iop[3:])
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-6:
        raise ValueError("ImageOrientationPatient has an invalid slice normal.")
    return ("Sagittal", "Coronal", "Axial")[int(np.argmax(np.abs(normal)))]


def _read_metadata_worker(file_path: str, backend: str, pydicom, dicomsdl):
    try:
        if backend == "dicomsdl":
            dcm = dicomsdl.open(file_path)

            def getter(key, default=None):
                return getattr(dcm, key, default)

        else:
            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
            getter = dcm.get
        return {
            "path": file_path,
            "InstanceNumber": getter("InstanceNumber", None),
            "ImagePositionPatient": getter("ImagePositionPatient", None),
            "ImageOrientationPatient": getter("ImageOrientationPatient", None),
            "PixelSpacing": getter("PixelSpacing", None),
            "Rows": getter("Rows", None),
            "Columns": getter("Columns", None),
            "SeriesInstanceUID": getter("SeriesInstanceUID", None),
            "Modality": getter("Modality", None),
            "ImageType": getter("ImageType", None),
            "RescaleSlope": getter("RescaleSlope", None),
            "RescaleIntercept": getter("RescaleIntercept", None),
            "NumberOfFrames": getter("NumberOfFrames", None),
            "PixelRepresentation": getter("PixelRepresentation", 0),
        }
    except Exception:
        return None


def _filter_valid_slices(metadata_list: list[dict], invalid_files: list[str]):
    valid_uids = [m["SeriesInstanceUID"] for m in metadata_list if m["SeriesInstanceUID"]]
    if not valid_uids:
        raise ValueError("No DICOM files had SeriesInstanceUID.")
    primary_uid = Counter(valid_uids).most_common(1)[0][0]

    valid_slices = []
    for meta in metadata_list:
        image_type = str(meta["ImageType"]).upper() if meta["ImageType"] else ""
        frames = meta["NumberOfFrames"]
        invalid = (
            (frames is not None and frames > 1)
            or "LOCALIZER" in image_type
            or "SCOUT" in image_type
            or meta["SeriesInstanceUID"] != primary_uid
            or meta["ImagePositionPatient"] is None
            or meta["ImageOrientationPatient"] is None
            or meta["PixelSpacing"] is None
            or meta["Rows"] is None
            or meta["Columns"] is None
        )
        if invalid:
            invalid_files.append(meta["path"])
        else:
            valid_slices.append(meta)
    if len(valid_slices) < 2:
        raise ValueError("Volumetric DICOM loading requires at least 2 valid slices.")

    planes = []
    for meta in valid_slices:
        iop = np.asarray(meta["ImageOrientationPatient"], dtype=float)
        planes.append(np.argmax(np.abs(np.cross(iop[:3], iop[3:]))))
    primary_plane = Counter(planes).most_common(1)[0][0]
    filtered = []
    for meta, plane in zip(valid_slices, planes):
        if plane == primary_plane:
            filtered.append(meta)
        else:
            invalid_files.append(meta["path"])
    if len(filtered) < 2:
        raise ValueError("Fewer than 2 valid slices after orientation filtering.")
    return filtered, invalid_files


def _sort_and_dedupe_slices(
    valid_slices: list[dict], invalid_files: list[str], sort_by_instance: bool
) -> list[dict]:
    iop = np.asarray(valid_slices[0]["ImageOrientationPatient"], dtype=float)
    normal_vector = np.cross(iop[:3], iop[3:])
    for meta in valid_slices:
        meta["z_location"] = float(
            np.dot(np.asarray(meta["ImagePositionPatient"], dtype=float), normal_vector)
        )

    if sort_by_instance:
        if any(meta["InstanceNumber"] is None for meta in valid_slices):
            raise ValueError("Cannot sort by InstanceNumber when any slice is missing it.")
        valid_slices.sort(key=lambda meta: meta["InstanceNumber"])
    else:
        valid_slices.sort(key=lambda meta: meta["z_location"])

    unique = []
    seen = set()
    for meta in valid_slices:
        rounded = round(meta["z_location"], 3)
        if rounded in seen:
            invalid_files.append(meta["path"])
            continue
        unique.append(meta)
        seen.add(rounded)
    if len(unique) < 2:
        raise ValueError("Fewer than 2 valid slices after duplicate removal.")
    return unique


def _read_pixel_worker(args):
    meta, backend, pydicom, dicomsdl, rescale, target_shape, fix_method = args
    try:
        if backend == "dicomsdl":
            dcm = dicomsdl.open(meta["path"])
            arr = dcm.pixelData(storedvalue=True)
            if meta.get("PixelRepresentation", 0) == 1 and arr.dtype == np.uint16:
                arr = arr.view(np.int16)
        else:
            dcm = pydicom.dcmread(meta["path"])
            arr = dcm.pixel_array
        arr = arr.astype(np.float32)
        if rescale and meta["RescaleSlope"] is not None and meta["RescaleIntercept"] is not None:
            arr = arr * float(meta["RescaleSlope"]) + float(meta["RescaleIntercept"])
        if arr.shape != target_shape:
            arr = _process_shape(arr, target_shape, fix_method)
        return arr
    except Exception as e:
        print(f"Error reading pixel data {meta['path']}: {e}")
        return None


def _process_shape(img: np.ndarray, target_shape: tuple[int, int], method: str) -> np.ndarray:
    if img.shape == target_shape:
        return img
    if method == "resize":
        return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    if method != "crop_pad":
        raise ValueError("fix_unequal_shapes_method must be 'crop_pad' or 'resize'.")

    result = np.min(img) * np.ones(target_shape, dtype=img.dtype)
    h, w = img.shape
    th, tw = target_shape
    h0 = max(0, h // 2 - th // 2)
    w0 = max(0, w // 2 - tw // 2)
    h1 = min(h, h0 + th)
    w1 = min(w, w0 + tw)
    out_h0 = max(0, th // 2 - h // 2)
    out_w0 = max(0, tw // 2 - w // 2)
    result[out_h0 : out_h0 + (h1 - h0), out_w0 : out_w0 + (w1 - w0)] = img[h0:h1, w0:w1]
    return result


def _series_geometry(valid_slices: list[dict]):
    pixel_spacing = valid_slices[0]["PixelSpacing"]
    dy, dx = float(pixel_spacing[0]), float(pixel_spacing[1])
    dz = abs(valid_slices[-1]["z_location"] - valid_slices[0]["z_location"]) / (
        len(valid_slices) - 1
    )

    iop = np.asarray(valid_slices[0]["ImageOrientationPatient"], dtype=float)
    row_cos = iop[:3]
    col_cos = iop[3:]
    ipp_first = np.asarray(valid_slices[0]["ImagePositionPatient"], dtype=float)
    ipp_last = np.asarray(valid_slices[-1]["ImagePositionPatient"], dtype=float)
    slice_vec = ipp_last - ipp_first
    if np.linalg.norm(slice_vec) > 1e-3:
        slice_vec /= np.linalg.norm(slice_vec)
    else:
        slice_vec = np.cross(row_cos, col_cos)

    orientation_code = _get_orientation_code_from_vectors(slice_vec, col_cos, row_cos)
    all_ipps = np.asarray([m["ImagePositionPatient"] for m in valid_slices], dtype=np.float32)
    return (dz, dy, dx), orientation_code, all_ipps


def _get_closest_axis(vector):
    idx = np.argmax(np.abs(vector))
    sign = 1 if vector[idx] > 0 else -1
    return idx, sign


def _get_orientation_code_from_vectors(v_d, v_h, v_w):
    labels = [[("R", "L"), ("A", "P"), ("I", "S")][idx] for idx in range(3)]
    code = ""
    for vector in [v_d, v_h, v_w]:
        idx, sign = _get_closest_axis(vector)
        code += labels[idx][1 if sign > 0 else 0]
    return code


def _reorient_array(image, spacing, current_code, target_code):
    axis_map = {
        "L": (0, 1),
        "R": (0, -1),
        "P": (1, 1),
        "A": (1, -1),
        "S": (2, 1),
        "I": (2, -1),
    }
    curr_axes = [axis_map[code] for code in current_code]
    target_axes = [axis_map[code] for code in target_code]
    perm = []
    flip = []
    for target_axis, target_sign in target_axes:
        for idx, (current_axis, current_sign) in enumerate(curr_axes):
            if current_axis == target_axis:
                perm.append(idx)
                flip.append(current_sign != target_sign)
                break
        else:
            raise ValueError(f"Cannot reorient from {current_code} to {target_code}.")

    image = np.transpose(image, perm)
    new_spacing = tuple(spacing[idx] for idx in perm)
    for axis, should_flip in enumerate(flip):
        if should_flip:
            image = np.flip(image, axis=axis)
    return np.ascontiguousarray(image), new_spacing
