"""RSNA knee PNG series dataset with explicit train/validation series scopes.

Each item is one MRI series. Validation therefore emits every eligible series;
the RSNA knee metric averages series probabilities to one prediction per study.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset


STATUS_VALUE = {
    "negative": 0.0,
    "not_mentioned": 0.0,
    "uncertain": 0.5,
    "positive": 1.0,
}
COMPETITION_LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def select_studies_for_mode(
    studies: pd.DataFrame,
    *,
    mode: str,
    fold: int,
    train_report_languages: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Apply the immutable CV split and an optional train-only language filter."""
    if mode == "train":
        selected = studies[
            (studies["always_validation"] == 0) & (studies["cv_fold"] != int(fold))
        ]
        if train_report_languages is not None:
            if isinstance(train_report_languages, str):
                raise TypeError(
                    "train_report_languages must be a sequence, not a string"
                )
            languages = tuple(
                dict.fromkeys(str(value) for value in train_report_languages)
            )
            if not languages or any(not value for value in languages):
                raise ValueError("train_report_languages must contain non-empty codes")
            if "language" not in selected.columns:
                raise ValueError(
                    "train_report_languages requires a language column in folds_csv"
                )
            selected = selected[selected["language"].isin(languages)]
    elif mode == "val":
        # Deliberately do not apply the training language restriction here:
        # validation must retain the rotating fold plus every expert study.
        selected = studies[
            (studies["always_validation"] == 1) | (studies["cv_fold"] == int(fold))
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if selected.empty:
        raise RuntimeError(f"No studies selected for mode={mode}, fold={fold}")
    return selected.copy()


def select_preferred_sequence_classes(
    series: pd.DataFrame,
    sequence_metadata: pd.DataFrame,
    preferred_classes: list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Prefer selected sequence classes per study without dropping studies.

    If a study has one or more completed series in a preferred class, only
    those series remain eligible. Otherwise every originally eligible series
    for that study is retained as an explicit fallback.
    """
    if isinstance(preferred_classes, str):
        raise TypeError("preferred sequence classes must be a sequence, not a string")
    preferred = tuple(dict.fromkeys(str(value) for value in preferred_classes))
    if not preferred or any(not value for value in preferred):
        raise ValueError("preferred sequence classes must contain non-empty values")
    keys = ["StudyInstanceUID", "SeriesInstanceUID"]
    required = {*keys, "sequence_class"}
    missing = required - set(sequence_metadata.columns)
    if missing:
        raise ValueError(f"Sequence metadata missing columns: {sorted(missing)}")
    if sequence_metadata[keys].duplicated().any():
        raise ValueError("Sequence metadata contains duplicate study/series rows")
    if series[keys].duplicated().any():
        raise ValueError("Eligible series table contains duplicate study/series rows")

    metadata = sequence_metadata[[*keys, "sequence_class"]].copy()
    merged = series.merge(metadata, on=keys, how="left", validate="one_to_one")
    if merged["sequence_class"].isna().any():
        row = merged.loc[merged["sequence_class"].isna(), keys].iloc[0]
        raise ValueError(
            "Sequence metadata missing eligible series: "
            f"{row.StudyInstanceUID}/{row.SeriesInstanceUID}"
        )
    is_preferred = merged["sequence_class"].isin(preferred)
    has_preferred = is_preferred.groupby(merged["StudyInstanceUID"]).transform("any")
    selected = merged[is_preferred | ~has_preferred].copy()
    study_count = int(merged["StudyInstanceUID"].nunique())
    preferred_studies = int(merged.loc[is_preferred, "StudyInstanceUID"].nunique())
    return selected, {
        "studies": study_count,
        "preferred_studies": preferred_studies,
        "fallback_studies": study_count - preferred_studies,
        "input_series": len(merged),
        "selected_series": len(selected),
    }


def resize_and_pad(image: np.ndarray, size: int) -> np.ndarray:
    """Preserve aspect ratio, resize, then center-pad to a square."""
    if image.ndim != 2 or not image.size:
        raise ValueError(f"Expected a non-empty 2D image, got {image.shape}")
    height, width = image.shape
    scale = min(size / height, size / width)
    resized_height = max(1, min(size, round(height * scale)))
    resized_width = max(1, min(size, round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )
    output = np.zeros((size, size), dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output


def physical_center_crop_and_resize(
    image: np.ndarray,
    *,
    pixel_spacing: tuple[float, float],
    fov_mm: float,
    size: int,
    center_pixel: tuple[float, float] | None = None,
) -> np.ndarray:
    """Crop/pad a square physical FOV about a pixel coordinate, then resize.

    ``pixel_spacing`` follows DICOM's row, column convention in mm/pixel.
    ``center_pixel`` follows DICOM's column, row convention. When omitted, the
    matrix center is used to preserve the legacy preprocessing exactly.
    Padding is performed in native pixel space so the output always represents
    exactly ``fov_mm`` along both in-plane patient dimensions.
    """
    if image.ndim != 2 or not image.size:
        raise ValueError(f"Expected a non-empty 2D image, got {image.shape}")
    row_spacing, column_spacing = (float(value) for value in pixel_spacing)
    if not all(
        math.isfinite(value) and value > 0
        for value in (row_spacing, column_spacing, fov_mm)
    ):
        raise ValueError(
            "Pixel spacing and physical FOV must be finite and positive, got "
            f"spacing={pixel_spacing}, fov_mm={fov_mm}"
        )
    if size <= 0:
        raise ValueError(f"Output size must be positive, got {size}")

    crop_height = max(1, round(fov_mm / row_spacing))
    crop_width = max(1, round(fov_mm / column_spacing))
    output = np.zeros((crop_height, crop_width), dtype=image.dtype)

    source_height, source_width = image.shape
    if center_pixel is None:
        # Retain the original integer centering convention byte-for-byte for
        # every pre-existing configuration.
        copy_height = min(source_height, crop_height)
        copy_width = min(source_width, crop_width)
        source_top = (source_height - copy_height) // 2
        source_left = (source_width - copy_width) // 2
        target_top = (crop_height - copy_height) // 2
        target_left = (crop_width - copy_width) // 2
    else:
        center_column, center_row = (float(value) for value in center_pixel)
        if not all(math.isfinite(value) for value in (center_column, center_row)):
            raise ValueError(f"Crop center must be finite, got {center_pixel}")
        requested_top = round(center_row - (crop_height - 1) / 2)
        requested_left = round(center_column - (crop_width - 1) / 2)
        source_top = max(0, requested_top)
        source_left = max(0, requested_left)
        source_bottom = min(source_height, requested_top + crop_height)
        source_right = min(source_width, requested_left + crop_width)
        copy_height = max(0, source_bottom - source_top)
        copy_width = max(0, source_right - source_left)
        target_top = source_top - requested_top
        target_left = source_left - requested_left
    output[
        target_top : target_top + copy_height,
        target_left : target_left + copy_width,
    ] = image[
        source_top : source_top + copy_height,
        source_left : source_left + copy_width,
    ]

    interpolation = (
        cv2.INTER_AREA if crop_height > size or crop_width > size else cv2.INTER_LINEAR
    )
    return cv2.resize(output, (size, size), interpolation=interpolation)


def resample_depth(volume: torch.Tensor, depth: int) -> torch.Tensor:
    """Resample ``(D,H,W)`` using linear interpolation only when downsampling."""
    if volume.ndim != 3 or volume.shape[0] == 0:
        raise ValueError(f"Expected a non-empty (D,H,W) volume, got {volume.shape}")
    if volume.shape[0] == depth:
        return volume
    value = volume[None, None]
    if volume.shape[0] > depth:
        value = F.interpolate(
            value,
            size=(depth, volume.shape[1], volume.shape[2]),
            mode="trilinear",
            align_corners=False,
        )
    else:
        value = F.interpolate(
            value,
            size=(depth, volume.shape[1], volume.shape[2]),
            mode="nearest",
        )
    return value[0, 0]


def stack_adjacent_slices(volume: torch.Tensor) -> torch.Tensor:
    """Return overlapping previous/current/next channels with replicated edges."""
    if volume.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {volume.shape}")
    indices = torch.arange(volume.shape[0], device=volume.device)
    previous = (indices - 1).clamp_min(0)
    following = (indices + 1).clamp_max(volume.shape[0] - 1)
    return torch.stack((volume[previous], volume, volume[following]), dim=1)


def make_slice_instances(
    volume: torch.Tensor,
    num_input_channels: int,
    position_volume: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a resampled volume into single-slice or adjacent-slice instances."""
    if num_input_channels == 1:
        return volume.unsqueeze(1)
    if num_input_channels == 3:
        return stack_adjacent_slices(volume)
    if num_input_channels == 4:
        if position_volume is None or position_volume.shape != volume.shape:
            raise ValueError(
                "num_input_channels=4 requires position_volume matching volume; "
                f"got volume={volume.shape}, position={getattr(position_volume, 'shape', None)}"
            )
        # Image channels retain their raw [0,255] scale. The signed position
        # field maps -1 medial, 0 midline/padding, +1 lateral to [0,255] so the
        # model's mean=std=0.5 normalization restores the signed coordinate.
        encoded_position = position_volume.clamp(-1, 1).add(1).mul(127.5)
        return torch.cat(
            (stack_adjacent_slices(volume), encoded_position.unsqueeze(1)), dim=1
        )
    raise ValueError(
        "RSNA knee MIL inputs support num_input_channels=1, 3, or 4, got "
        f"{num_input_channels}"
    )


def medial_lateral_position_volume(
    shapes: list[tuple[int, int]],
    *,
    depth: int,
    image_size: int,
    axis: str,
    sign: int,
) -> torch.Tensor:
    """Build a signed coordinate aligned with native slices and resize padding."""
    if not shapes or any(height <= 0 or width <= 0 for height, width in shapes):
        raise ValueError("Native image shapes must be non-empty and positive")
    if sign not in {-1, 1}:
        raise ValueError(f"position sign must be -1 or 1, got {sign}")
    if axis == "depth":
        native = torch.linspace(-1.0, 1.0, len(shapes)).view(-1, 1, 1)
        native = native.expand(-1, image_size, image_size).mul(sign)
        return resample_depth(native, depth)
    if axis != "column":
        raise ValueError(f"position axis must be column or depth, got {axis!r}")

    maps = []
    for height, width in shapes:
        scale = min(image_size / height, image_size / width)
        resized_height = max(1, min(image_size, round(height * scale)))
        resized_width = max(1, min(image_size, round(width * scale)))
        row = torch.linspace(-1.0, 1.0, resized_width).mul(sign)
        value = torch.zeros((image_size, image_size), dtype=torch.float32)
        top = (image_size - resized_height) // 2
        left = (image_size - resized_width) // 2
        value[top : top + resized_height, left : left + resized_width] = row.expand(
            resized_height, -1
        )
        maps.append(value)
    return resample_depth(torch.stack(maps), depth)


def canonicalize_medial_lateral(
    volume: torch.Tensor, *, axis: str, sign: int
) -> torch.Tensor:
    """Orient a volume so array position always increases medial to lateral."""
    if volume.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {volume.shape}")
    if sign not in {-1, 1}:
        raise ValueError(f"position sign must be -1 or 1, got {sign}")
    if axis not in {"column", "depth"}:
        raise ValueError(f"position axis must be column or depth, got {axis!r}")
    if sign == 1:
        return volume
    return volume.flip(-1 if axis == "column" else 0)


def crop_medial_lateral_compartment(
    volume: torch.Tensor,
    *,
    axis: str,
    compartment: str,
    fraction: float,
) -> torch.Tensor:
    """Crop a canonical medial-to-lateral volume and restore its input shape.

    ``volume`` must already be oriented so its anatomical axis increases from
    medial to lateral. Axial/coronal series encode that axis in columns, while
    sagittal series encode it across depth.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {volume.shape}")
    if axis not in {"column", "depth"}:
        raise ValueError(f"position axis must be column or depth, got {axis!r}")
    if compartment not in {"medial", "lateral"}:
        raise ValueError(
            f"medial_lateral_compartment must be medial or lateral, got {compartment!r}"
        )
    if not math.isfinite(fraction) or not 0.5 < fraction <= 1.0:
        raise ValueError(
            f"medial_lateral_compartment_fraction must be in (0.5, 1], got {fraction!r}"
        )

    if axis == "depth":
        extent = max(1, round(volume.shape[0] * fraction))
        cropped = volume[:extent] if compartment == "medial" else volume[-extent:]
        return resample_depth(cropped, volume.shape[0])

    extent = max(1, round(volume.shape[-1] * fraction))
    cropped = volume[..., :extent] if compartment == "medial" else volume[..., -extent:]
    return F.interpolate(
        cropped[:, None],
        size=volume.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def augment_volume(volume: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
    """Apply flip-free, volume-consistent spatial and intensity augmentation.

    There is deliberately no axis permutation, reflection, or depth transform.
    One positive-determinant affine matrix is shared by every depth slice.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {volume.shape}")
    probability = float(params["probability"])
    if torch.rand(()) >= probability:
        return volume

    angle = math.radians(
        float(
            torch.empty(()).uniform_(
                -params["rotate_degrees"], params["rotate_degrees"]
            )
        )
    )
    scale = float(torch.empty(()).uniform_(params["scale_min"], params["scale_max"]))
    translation = float(params["translate_fraction"])
    tx = float(torch.empty(()).uniform_(-translation, translation)) * 2.0
    ty = float(torch.empty(()).uniform_(-translation, translation)) * 2.0
    cosine = math.cos(angle) / scale
    sine = math.sin(angle) / scale
    theta = volume.new_tensor([[cosine, -sine, tx], [sine, cosine, ty]])
    batch = volume[:, None]
    grid = F.affine_grid(
        theta[None].expand(volume.shape[0], -1, -1),
        batch.shape,
        align_corners=False,
    )
    augmented = F.grid_sample(
        batch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0]

    augmented = augmented.div(255.0).clamp(0, 1)
    gamma = float(torch.empty(()).uniform_(params["gamma_min"], params["gamma_max"]))
    contrast = float(
        torch.empty(()).uniform_(params["contrast_min"], params["contrast_max"])
    )
    brightness = float(params["brightness_delta"])
    brightness = float(torch.empty(()).uniform_(-brightness, brightness))
    augmented = augmented.pow(gamma)
    augmented = (augmented - 0.5) * contrast + 0.5 + brightness
    noise_std = float(torch.empty(()).uniform_(0, params["noise_std_max"]))
    if noise_std > 0:
        augmented = augmented + torch.randn_like(augmented) * noise_std
    return augmented.clamp_(0, 1).mul_(255.0)


def augment_volume_with_position(
    volume: torch.Tensor,
    position: torch.Tensor,
    params: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one sampled spatial affine to image and signed position volumes.

    Intensity augmentation is applied only to the image. Coordinate padding is
    zero, representing the neutral midline rather than either anatomical side.
    """
    if volume.shape != position.shape or volume.ndim != 3:
        raise ValueError(
            f"Image and position volumes must share (D,H,W), got {volume.shape} "
            f"and {position.shape}"
        )
    probability = float(params["probability"])
    if torch.rand(()) >= probability:
        return volume, position

    angle = math.radians(
        float(
            torch.empty(()).uniform_(
                -params["rotate_degrees"], params["rotate_degrees"]
            )
        )
    )
    scale = float(torch.empty(()).uniform_(params["scale_min"], params["scale_max"]))
    translation = float(params["translate_fraction"])
    tx = float(torch.empty(()).uniform_(-translation, translation)) * 2.0
    ty = float(torch.empty(()).uniform_(-translation, translation)) * 2.0
    cosine = math.cos(angle) / scale
    sine = math.sin(angle) / scale
    theta = volume.new_tensor([[cosine, -sine, tx], [sine, cosine, ty]])
    grid = F.affine_grid(
        theta[None].expand(volume.shape[0], -1, -1),
        (volume.shape[0], 1, volume.shape[1], volume.shape[2]),
        align_corners=False,
    )
    image = F.grid_sample(
        volume[:, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0]
    position = F.grid_sample(
        position[:, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0].clamp_(-1, 1)

    image = image.div(255.0).clamp(0, 1)
    gamma = float(torch.empty(()).uniform_(params["gamma_min"], params["gamma_max"]))
    contrast = float(
        torch.empty(()).uniform_(params["contrast_min"], params["contrast_max"])
    )
    brightness = float(params["brightness_delta"])
    brightness = float(torch.empty(()).uniform_(-brightness, brightness))
    image = image.pow(gamma)
    image = (image - 0.5) * contrast + 0.5 + brightness
    noise_std = float(torch.empty(()).uniform_(0, params["noise_std_max"]))
    if noise_std > 0:
        image = image + torch.randn_like(image) * noise_std
    return image.clamp_(0, 1).mul_(255.0), position


def _read_extractions(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            uid = record["StudyInstanceUID"]
            if uid in records:
                raise ValueError(
                    f"Duplicate extraction UID at line {line_number}: {uid}"
                )
            records[uid] = record["labels"]
    return records


def label_target(
    label_data: dict[str, Any],
    *,
    mode: str,
    uncertain_train_value: float | None = None,
) -> tuple[float, bool]:
    """Resolve a four-state or explicitly soft report label for one mode."""
    if "value" in label_data:
        value = float(label_data["value"])
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise ValueError(f"Invalid soft target {value!r}")
        return value, False
    status = label_data["status"]
    if status not in STATUS_VALUE:
        raise ValueError(f"Unknown status {status!r}")
    value = STATUS_VALUE[status]
    if mode == "train" and status == "uncertain" and uncertain_train_value is not None:
        value = float(uncertain_train_value)
        if value not in {0.0, 0.5, 1.0}:
            raise ValueError(
                "uncertain_train_value must be 0, 0.5, or 1, got "
                f"{uncertain_train_value!r}"
            )
    return value, status == "uncertain"


class Dataset(TorchDataset):
    """One converted series per item under a mode-specific eligibility scope."""

    def __init__(self, cfg: Any, mode: str):
        if mode not in {"train", "val"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.cfg = cfg
        self.mode = mode
        self.class_names = cfg.require("class_names")
        self.image_size = int(cfg.require("image_size"))
        self.depth = int(cfg.require("depth"))
        self.num_input_channels = int(cfg.get("num_input_channels", 3) or 3)
        if self.num_input_channels not in {1, 3, 4}:
            raise ValueError(
                "RSNA knee MIL inputs support num_input_channels=1, 3, or 4, got "
                f"{self.num_input_channels}"
            )
        self.add_medial_lateral_channel = bool(
            cfg.get("add_medial_lateral_channel", False)
        )
        if self.add_medial_lateral_channel != (self.num_input_channels == 4):
            raise ValueError(
                "add_medial_lateral_channel must be true exactly when "
                "num_input_channels=4"
            )
        self.canonicalize_medial_lateral = bool(
            cfg.get("canonicalize_medial_lateral", False)
        )
        self.return_series_geometry = bool(cfg.get("return_series_geometry", False))
        if self.add_medial_lateral_channel and self.canonicalize_medial_lateral:
            raise ValueError(
                "Position-channel and canonical-reorientation ablations are mutually exclusive"
            )
        self.medial_lateral_compartment = cfg.get("medial_lateral_compartment")
        self.medial_lateral_compartment_fraction = float(
            cfg.get("medial_lateral_compartment_fraction", 0.65) or 0.65
        )
        if self.medial_lateral_compartment is not None:
            if not self.canonicalize_medial_lateral:
                raise ValueError(
                    "medial_lateral_compartment requires canonicalize_medial_lateral"
                )
            if self.medial_lateral_compartment not in {"medial", "lateral"}:
                raise ValueError("medial_lateral_compartment must be medial or lateral")
            if not 0.5 < self.medial_lateral_compartment_fraction <= 1.0:
                raise ValueError(
                    "medial_lateral_compartment_fraction must be in (0.5, 1]"
                )
        self.physical_fov_mm = cfg.get("physical_fov_mm")
        self.physical_fov_center_csv = cfg.get("physical_fov_center_csv")
        if self.physical_fov_mm is not None:
            self.physical_fov_mm = float(self.physical_fov_mm)
            if not math.isfinite(self.physical_fov_mm) or self.physical_fov_mm <= 0:
                raise ValueError(
                    f"physical_fov_mm must be finite and positive, got {self.physical_fov_mm}"
                )
            if self.add_medial_lateral_channel:
                raise ValueError(
                    "Physical-FOV cropping is not yet aligned with the medial/lateral channel"
                )
        elif self.physical_fov_center_csv:
            raise ValueError(
                "physical_fov_center_csv requires physical_fov_mm to be configured"
            )
        self.png_root = Path(cfg.require("png_root"))

        studies = pd.read_csv(cfg.require("train_csv"))
        series = pd.read_csv(cfg.require("train_series_csv"))
        folds = pd.read_csv(cfg.require("folds_csv"))
        mode_labels_key = f"{mode}_labels_jsonl"
        labels_jsonl = cfg.get(mode_labels_key) or cfg.require("labels_jsonl")
        extracted = _read_extractions(Path(labels_jsonl))
        train_label_masks: dict[str, list[bool]] | None = None
        if mode == "train" and cfg.get("train_label_mask_json"):
            mask_path = Path(cfg.require("train_label_mask_json"))
            mask_payload = json.loads(mask_path.read_text(encoding="utf-8"))
            if mask_payload.get("class_names") != self.class_names:
                raise ValueError("Training label-mask class names do not match config")
            train_label_masks = mask_payload.get("study_masks")
            if not isinstance(train_label_masks, dict):
                raise ValueError("Training label-mask artifact requires study_masks")
        train_target_policy: dict[str, Any] | None = None
        policy_key = f"{mode}_target_policy_json"
        if mode in ("train", "val") and cfg.get(policy_key):
            policy_path = Path(cfg.require(policy_key))
            policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
            if policy_payload.get("class_names") != self.class_names:
                raise ValueError(
                    "Training target-policy class names do not match config"
                )
            targets = policy_payload.get("study_targets")
            weights = policy_payload.get("study_weights")
            if not isinstance(targets, dict) or not isinstance(weights, dict):
                raise ValueError(
                    "Training target-policy artifact requires study_targets and study_weights"
                )
            if set(targets) != set(weights):
                raise ValueError(
                    "Training target-policy target/weight coverage differs"
                )
            train_target_policy = {"targets": targets, "weights": weights}
        train_target_labels = None
        train_aux_target_labels: list[str] = []
        train_aux_target_weight = 0.0
        if mode == "train" and cfg.get("train_target_labels"):
            train_target_labels = cfg.require("train_target_labels")
            if (
                not isinstance(train_target_labels, list)
                or not train_target_labels
                or len(train_target_labels) != len(set(train_target_labels))
                or any(label not in self.class_names for label in train_target_labels)
            ):
                raise ValueError(
                    "train_target_labels must be a unique non-empty subset of class_names"
                )
            train_aux_target_labels = list(cfg.get("train_aux_target_labels", []) or [])
            if (
                len(train_aux_target_labels) != len(set(train_aux_target_labels))
                or any(
                    label not in self.class_names for label in train_aux_target_labels
                )
                or set(train_aux_target_labels) & set(train_target_labels)
            ):
                raise ValueError(
                    "train_aux_target_labels must be a unique subset of class_names "
                    "disjoint from train_target_labels"
                )
            train_aux_target_weight = float(
                cfg.get("train_aux_target_weight", 0.25) or 0.0
            )
            if train_aux_target_labels and not 0.0 < train_aux_target_weight <= 1.0:
                raise ValueError(
                    "train_aux_target_weight must be in (0, 1] when auxiliary "
                    f"targets are configured, got {train_aux_target_weight}"
                )
        studies = studies.merge(folds, on="StudyInstanceUID", validate="one_to_one")
        if len(studies) != len(folds) or len(extracted) != len(studies):
            raise ValueError(
                "Study, fold, and extraction artifacts must have exact coverage"
            )

        selected = select_studies_for_mode(
            studies,
            mode=mode,
            fold=int(cfg.fold),
            train_report_languages=cfg.get("train_report_languages"),
        )

        scope_key = "train_series_scope" if mode == "train" else "val_series_scope"
        series_scope = cfg.get(scope_key, "fluid_sensitive") or "fluid_sensitive"
        if series_scope == "fluid_sensitive":
            series = series[series["Fluid_Sensitive"] == 1].copy()
        elif series_scope == "non_fluid_sensitive":
            series = series[series["Fluid_Sensitive"] == 0].copy()
        elif series_scope == "all_completed":
            series = series.copy()
        else:
            raise ValueError(
                f"Unsupported {scope_key}={series_scope!r}; expected fluid_sensitive, "
                "non_fluid_sensitive, or all_completed"
            )
        series = series.merge(
            selected[["StudyInstanceUID"]],
            on="StudyInstanceUID",
            validate="many_to_one",
        )
        series["series_dir"] = series.apply(
            lambda row: (
                self.png_root
                / "train_series"
                / row.StudyInstanceUID
                / row.SeriesInstanceUID
            ),
            axis=1,
        )
        complete = series["series_dir"].map(
            lambda path: (path / "conversion.json").is_file()
        )
        self.skipped_incomplete_series = int((~complete).sum())
        series = series[complete].sort_values(["StudyInstanceUID", "SeriesInstanceUID"])
        if series.empty:
            raise RuntimeError(f"No completed {series_scope} PNG series for {mode}")

        preference_key = f"{mode}_preferred_sequence_classes"
        preferred_sequence_classes = cfg.get(preference_key)
        self.sequence_preference_audit = None
        if preferred_sequence_classes:
            metadata = pd.read_csv(
                cfg.require("sequence_metadata_csv"), low_memory=False
            )
            if "split" in metadata.columns:
                metadata = metadata[metadata["split"] == "train"].copy()
            series, self.sequence_preference_audit = select_preferred_sequence_classes(
                series, metadata, preferred_sequence_classes
            )

        selected_uids = set(selected["StudyInstanceUID"])
        present_uids = set(series["StudyInstanceUID"])
        missing = sorted(selected_uids - present_uids)
        if missing:
            raise RuntimeError(
                f"{len(missing)} selected studies have no completed {series_scope} series; "
                f"first={missing[0]}"
            )

        all_uids = sorted(studies["StudyInstanceUID"].tolist())
        self.study_to_index = {uid: index for index, uid in enumerate(all_uids)}
        official = studies.set_index("StudyInstanceUID")
        physical_spacing_by_series = {}
        physical_center_by_series = {}
        if self.physical_fov_mm is not None:
            spacing = pd.read_csv(cfg.require("physical_fov_metadata_csv"))
            if "split" in spacing.columns:
                spacing = spacing[spacing["split"] == "train"].copy()
            required = {
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "PixelSpacingRow",
                "PixelSpacingColumn",
            }
            if required - set(spacing.columns):
                raise ValueError(
                    "Physical-FOV metadata missing columns: "
                    f"{sorted(required - set(spacing.columns))}"
                )
            if spacing[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().any():
                raise ValueError("Duplicate series in physical-FOV metadata")
            physical_spacing_by_series = spacing.set_index("SeriesInstanceUID").to_dict(
                "index"
            )
            if self.physical_fov_center_csv:
                centers = pd.read_csv(self.physical_fov_center_csv, low_memory=False)
                if "split" in centers.columns:
                    centers = centers[centers["split"] == "train"].copy()
                required = {
                    "StudyInstanceUID",
                    "SeriesInstanceUID",
                    "coordinate_status",
                    "center_pixel_column",
                    "center_pixel_row",
                }
                if required - set(centers.columns):
                    raise ValueError(
                        "Physical-FOV center metadata missing columns: "
                        f"{sorted(required - set(centers.columns))}"
                    )
                if centers[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().any():
                    raise ValueError("Duplicate series in physical-FOV center metadata")
                centers = centers[centers["coordinate_status"] == "ok"].copy()
                physical_center_by_series = centers.set_index(
                    "SeriesInstanceUID"
                ).to_dict("index")
        position_by_series = {}
        if (
            self.add_medial_lateral_channel
            or self.canonicalize_medial_lateral
            or self.return_series_geometry
        ):
            position = pd.read_csv(cfg.require("medial_lateral_position_csv"))
            required = {
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "position_axis",
                "position_sign",
                "predicted_laterality",
            }
            if required - set(position.columns):
                raise ValueError(
                    "Medial/lateral position metadata missing columns: "
                    f"{sorted(required - set(position.columns))}"
                )
            if position[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().any():
                raise ValueError("Duplicate series in medial/lateral position metadata")
            position_by_series = position.set_index("SeriesInstanceUID").to_dict(
                "index"
            )
        self.items: list[dict[str, Any]] = []
        plane_to_index = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
        for row in series.itertuples(index=False):
            uid = row.StudyInstanceUID
            extraction = extracted[uid]
            values = []
            uncertain = []
            for label in self.class_names:
                label_data = extraction[label]
                try:
                    value, is_uncertain = label_target(
                        label_data,
                        mode=mode,
                        uncertain_train_value=cfg.get("uncertain_train_value"),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid target for {uid}/{label}: {exc}"
                    ) from exc
                values.append(value)
                uncertain.append(is_uncertain)
            is_expert = bool(official.loc[uid, "expert_labeled"])
            if is_expert:
                for label in COMPETITION_LABELS:
                    index = self.class_names.index(label)
                    value = official.loc[uid, label]
                    if pd.isna(value) or value not in (0, 1):
                        raise ValueError(
                            f"Invalid expert label for {uid}/{label}: {value}"
                        )
                    values[index] = float(value)
                    uncertain[index] = False
            target_weights = [1.0] * len(self.class_names)
            if train_target_policy is not None:
                if uid not in train_target_policy["targets"]:
                    raise ValueError(f"Training target policy missing study {uid}")
                values = train_target_policy["targets"][uid]
                target_weights = train_target_policy["weights"][uid]
                if (
                    len(values) != len(self.class_names)
                    or len(target_weights) != len(self.class_names)
                    or not all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and 0 <= float(value) <= 1
                        for value in values
                    )
                    or not all(
                        isinstance(weight, (int, float))
                        and not isinstance(weight, bool)
                        and math.isfinite(float(weight))
                        and 0 <= float(weight) <= 1
                        for weight in target_weights
                    )
                ):
                    raise ValueError(f"Invalid training target policy for {uid}")
                values = list(map(float, values))
                target_weights = list(map(float, target_weights))
                if mode == "val":
                    uncertain = [False] * len(values)
                    if is_expert:
                        for label in COMPETITION_LABELS:
                            values[self.class_names.index(label)] = float(official.loc[uid, label])
            item = {
                "study_uid": uid,
                "series_uid": row.SeriesInstanceUID,
                "series_dir": Path(row.series_dir),
                "study_index": self.study_to_index[uid],
                "is_expert": is_expert,
                "fluid_sensitive": bool(row.Fluid_Sensitive),
                "anatomical_plane_index": plane_to_index.get(row.Anatomical_Plane, -1),
                "sequence_class": getattr(row, "sequence_class", None),
                "y": values,
                "uncertain": uncertain,
                "y_weight": target_weights,
                "y_mask": (
                    train_label_masks[uid]
                    if train_label_masks is not None and uid in train_label_masks
                    else [True] * len(self.class_names)
                ),
            }
            if train_target_labels is not None:
                target_label_set = set(train_target_labels)
                auxiliary_label_set = set(train_aux_target_labels)
                item["y_mask"] = [
                    valid and label in target_label_set | auxiliary_label_set
                    for valid, label in zip(
                        item["y_mask"], self.class_names, strict=True
                    )
                ]
                item["y_weight"] = [
                    weight
                    * (
                        1.0
                        if label in target_label_set
                        else train_aux_target_weight
                        if label in auxiliary_label_set
                        else 0.0
                    )
                    for weight, label in zip(
                        item["y_weight"], self.class_names, strict=True
                    )
                ]
            if len(item["y_mask"]) != len(self.class_names) or not all(
                isinstance(value, bool) for value in item["y_mask"]
            ):
                raise ValueError(f"Invalid training label mask for {uid}")
            if self.physical_fov_mm is not None:
                if row.SeriesInstanceUID not in physical_spacing_by_series:
                    raise ValueError(
                        f"Missing physical-FOV metadata for {row.SeriesInstanceUID}"
                    )
                metadata = physical_spacing_by_series[row.SeriesInstanceUID]
                if metadata["StudyInstanceUID"] != uid:
                    raise ValueError(
                        f"Physical-FOV metadata study mismatch for {row.SeriesInstanceUID}"
                    )
                spacing_values = (
                    float(metadata["PixelSpacingRow"]),
                    float(metadata["PixelSpacingColumn"]),
                )
                if not all(
                    math.isfinite(value) and value > 0 for value in spacing_values
                ):
                    raise ValueError(
                        f"Invalid pixel spacing for {row.SeriesInstanceUID}: {spacing_values}"
                    )
                item["pixel_spacing"] = spacing_values
                if self.physical_fov_center_csv:
                    if row.SeriesInstanceUID not in physical_center_by_series:
                        raise ValueError(
                            "Missing predicted physical-FOV center for "
                            f"{row.SeriesInstanceUID}"
                        )
                    center_metadata = physical_center_by_series[row.SeriesInstanceUID]
                    if center_metadata["StudyInstanceUID"] != uid:
                        raise ValueError(
                            "Physical-FOV center study mismatch for "
                            f"{row.SeriesInstanceUID}"
                        )
                    center_pixel = (
                        float(center_metadata["center_pixel_column"]),
                        float(center_metadata["center_pixel_row"]),
                    )
                    if not all(math.isfinite(value) for value in center_pixel):
                        raise ValueError(
                            f"Invalid physical-FOV center for {row.SeriesInstanceUID}: "
                            f"{center_pixel}"
                        )
                    item["physical_fov_center_pixel"] = center_pixel
            if (
                self.add_medial_lateral_channel
                or self.canonicalize_medial_lateral
                or self.return_series_geometry
            ):
                if row.SeriesInstanceUID not in position_by_series:
                    raise ValueError(
                        f"Missing medial/lateral metadata for {row.SeriesInstanceUID}"
                    )
                metadata = position_by_series[row.SeriesInstanceUID]
                if metadata["StudyInstanceUID"] != uid:
                    raise ValueError(
                        f"Position metadata study mismatch for {row.SeriesInstanceUID}"
                    )
                item["position_axis"] = str(metadata["position_axis"])
                item["position_sign"] = int(metadata["position_sign"])
                item["predicted_laterality"] = str(metadata["predicted_laterality"])
                if row.Anatomical_Plane not in plane_to_index:
                    raise ValueError(
                        f"Unsupported anatomical plane for {row.SeriesInstanceUID}: "
                        f"{row.Anatomical_Plane!r}"
                    )
                item["anatomical_plane_index"] = plane_to_index[row.Anatomical_Plane]
            self.items.append(item)
        self.collate_fn = None

    def __len__(self) -> int:
        return len(self.items)

    def _load_volume(self, item: dict[str, Any]) -> torch.Tensor:
        series_dir = item["series_dir"]
        paths = sorted(series_dir.glob("*.png"))
        if not paths:
            raise RuntimeError(f"No PNG slices in {series_dir}")
        images = []
        shapes = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"OpenCV could not read {path}")
            shapes.append(image.shape)
            if self.physical_fov_mm is None:
                image = resize_and_pad(image, self.image_size)
            else:
                image = physical_center_crop_and_resize(
                    image,
                    pixel_spacing=item["pixel_spacing"],
                    fov_mm=self.physical_fov_mm,
                    size=self.image_size,
                    center_pixel=item.get("physical_fov_center_pixel"),
                )
            images.append(image)
        volume = torch.from_numpy(np.stack(images)).float()
        volume = resample_depth(volume, self.depth)
        if self.canonicalize_medial_lateral:
            volume = canonicalize_medial_lateral(
                volume,
                axis=item["position_axis"],
                sign=item["position_sign"],
            )
        if self.medial_lateral_compartment is not None:
            volume = crop_medial_lateral_compartment(
                volume,
                axis=item["position_axis"],
                compartment=self.medial_lateral_compartment,
                fraction=self.medial_lateral_compartment_fraction,
            )
        position = None
        if self.add_medial_lateral_channel:
            position = medial_lateral_position_volume(
                shapes,
                depth=self.depth,
                image_size=self.image_size,
                axis=item["position_axis"],
                sign=item["position_sign"],
            )
        augmentation = self.cfg.get("augmentation")
        if self.mode == "train" and augmentation and augmentation.get("enabled", False):
            if position is None:
                volume = augment_volume(volume, augmentation)
            else:
                volume, position = augment_volume_with_position(
                    volume, position, augmentation
                )
        return make_slice_instances(volume, self.num_input_channels, position)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        target = torch.tensor(item["y"], dtype=torch.float32)
        output = {
            "x": self._load_volume(item),
            "y": target,
            "y_mask": torch.tensor(item["y_mask"], dtype=torch.bool),
            "y_weight": torch.tensor(item["y_weight"], dtype=torch.float32),
            "uncertain_mask": torch.tensor(item["uncertain"], dtype=torch.bool),
            "is_expert": torch.tensor(item["is_expert"], dtype=torch.bool),
            "fluid_sensitive": torch.tensor(item["fluid_sensitive"], dtype=torch.bool),
            "study_index": torch.tensor(item["study_index"], dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.return_series_geometry:
            output.update(
                {
                    "position_axis": torch.tensor(
                        0 if item["position_axis"] == "depth" else 1,
                        dtype=torch.long,
                    ),
                    "position_sign": torch.tensor(
                        item["position_sign"], dtype=torch.float32
                    ),
                    "anatomical_plane": torch.tensor(
                        item["anatomical_plane_index"], dtype=torch.long
                    ),
                }
            )
        return output
