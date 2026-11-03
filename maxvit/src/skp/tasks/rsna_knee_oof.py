"""Atomic, fold-scoped exports of validation predictions already in memory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from skp.metrics.rsna_knee import ValidationMetrics, _metric_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".csv", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            frame.to_csv(stream, index=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".json", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _view_rows(
    metric: ValidationMetrics, payload: dict[str, torch.Tensor], view: str
) -> torch.Tensor:
    if view == "all_series":
        return torch.ones_like(payload["fluid_sensitive"], dtype=torch.bool)
    if view == "fluid_sensitive":
        return payload["fluid_sensitive"].bool()
    if view == "non_fluid_sensitive":
        return ~payload["fluid_sensitive"].bool()
    raise ValueError(f"Unknown validation series view: {view}")


def _study_frame(
    metric: ValidationMetrics,
    payload: dict[str, torch.Tensor],
    dataset: Any,
    view: str,
    *,
    has_auxiliary: bool,
    logged_metrics: dict[str, Any],
) -> pd.DataFrame:
    rows = _view_rows(metric, payload, view)
    if not rows.any():
        raise RuntimeError(f"OOF view has no validation series: {view}")
    index = payload["study_index"][rows].unique(sorted=True)
    planes = payload["anatomical_plane"][rows]
    allowed = (
        metric.allowed_label_planes.cpu() if metric.has_label_plane_filters else None
    )
    prediction, target, valid, uncertain, expert = metric._aggregate_studies(
        payload["logits"][rows],
        payload["target"][rows],
        payload["valid"][rows],
        payload["uncertain"][rows],
        payload["is_expert"][rows],
        payload["study_index"][rows],
        planes,
        allowed,
    )
    if not valid.all():
        raise RuntimeError("OOF export requires valid targets for every label")
    auxiliary = None
    if has_auxiliary:
        auxiliary = metric._aggregate_studies(
            payload["aux_logits"][rows],
            payload["target"][rows],
            payload["valid"][rows],
            payload["uncertain"][rows],
            payload["is_expert"][rows],
            payload["study_index"][rows],
            planes,
            allowed,
        )[0]
    uid_by_index = {number: uid for uid, number in dataset.study_to_index.items()}
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": [uid_by_index[int(value)] for value in index],
            "study_index": index.numpy(),
            "is_expert": expert.numpy().astype(bool),
            "n_validation_items": [
                int((payload["study_index"][rows] == value).sum()) for value in index
            ],
        }
    )
    for column, name in enumerate(metric.class_names):
        key = _metric_name(name)
        frame[f"target_{key}"] = target[:, column].numpy()
        frame[f"pred_{key}"] = prediction[:, column].numpy()
        if auxiliary is not None:
            frame[f"aux_pred_{key}"] = auxiliary[:, column].numpy()
        frame[f"uncertain_{key}"] = uncertain[:, column].numpy().astype(bool)
    # The saved study probabilities must reproduce the just-logged primary
    # validation macro, including label-specific plane fallback behavior.
    gold = frame["is_expert"].to_numpy()
    aucs = [
        roc_auc_score(
            frame.loc[gold, f"target_{_metric_name(name)}"],
            frame.loc[gold, f"pred_{_metric_name(name)}"],
        )
        for name in metric.evaluation_labels
    ]
    expected = float(logged_metrics[metric._key("expert", view, metric.macro_name)])
    if not np.isclose(np.mean(aucs), expected, rtol=0, atol=2e-6):
        raise RuntimeError(
            f"OOF study predictions do not reproduce {view} expert macro: "
            f"saved={np.mean(aucs):.8f}, logged={expected:.8f}"
        )
    return frame


def save_validation_oof(
    metric: ValidationMetrics,
    dataset: Any,
    cfg: Any,
    *,
    epoch: int,
    global_step: int,
    global_rank: int,
    prediction_weights: str,
    logged_metrics: dict[str, Any],
) -> Path | None:
    """Gather on every rank; write one complete epoch on rank zero only."""
    if prediction_weights not in {"ema", "raw"}:
        raise ValueError("prediction_weights must be ema or raw")
    payload = metric.gather_oof_series()
    if global_rank != 0:
        return None
    count = len(dataset.items)
    study_bags = ["series_indices" in item for item in dataset.items]
    if any(study_bags) and not all(study_bags):
        raise RuntimeError("Validation dataset mixes series and study-bag items")
    sample_unit = "study" if all(study_bags) else "series"
    indices = payload["series_index"].numpy()
    if len(indices) != count or not np.array_equal(np.sort(indices), np.arange(count)):
        raise RuntimeError(
            "OOF export requires exactly one prediction per validation item"
        )
    order = np.argsort(indices)
    payload = {name: value[order] for name, value in payload.items()}
    for position, item in enumerate(dataset.items):
        if int(payload["study_index"][position]) != int(item["study_index"]):
            raise RuntimeError(f"OOF item/study index mismatch at row {position}")
    aux_is_finite = torch.isfinite(payload["aux_logits"])
    if aux_is_finite.any() and not aux_is_finite.all():
        raise RuntimeError("Auxiliary validation logits are only partly available")
    has_auxiliary = bool(aux_is_finite.all())

    fold = cfg.get("fold")
    split = f"fold{fold}" if fold is not None else "fixed_split"
    parent = Path(cfg.save_dir) / split / "oof"
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / f"epoch={epoch:03d}-step={global_step:07d}"
    if directory.exists():
        raise FileExistsError(f"OOF epoch export already exists: {directory}")
    # Publish the directory only after every artifact and the manifest exist.
    # A crash leaves a clearly incomplete hidden temp directory, not a false
    # completed epoch; reruns may safely write the intended final path.
    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f".epoch={epoch:03d}-step={global_step:07d}.", dir=parent
        )
    )
    samples_path = work_dir / "samples.npz"
    arrays = {
        "sample_index": indices[order],
        "study_uid": np.asarray([item["study_uid"] for item in dataset.items]),
        "study_index": payload["study_index"].numpy(),
        "is_expert": payload["is_expert"].numpy(),
        "fluid_sensitive": payload["fluid_sensitive"].numpy(),
        "anatomical_plane": payload["anatomical_plane"].numpy(),
        "target": payload["target"].numpy(),
        "uncertain": payload["uncertain"].numpy(),
        "primary_probability": payload["logits"].sigmoid().numpy(),
        "class_names": np.asarray(metric.class_names),
    }
    if sample_unit == "series":
        arrays["series_uid"] = np.asarray(
            [item["series_uid"] for item in dataset.items]
        )
    if has_auxiliary:
        arrays["auxiliary_probability"] = payload["aux_logits"].sigmoid().numpy()
    _atomic_npz(samples_path, arrays)
    artifacts = {"samples.npz": _sha256(samples_path)}
    view_counts = {}
    for view in metric.series_views:
        frame = _study_frame(
            metric,
            payload,
            dataset,
            view,
            has_auxiliary=has_auxiliary,
            logged_metrics=logged_metrics,
        )
        path = work_dir / f"studies_{view}.csv"
        _atomic_csv(path, frame)
        artifacts[path.name] = _sha256(path)
        view_counts[view] = {
            "studies": len(frame),
            "expert": int(frame["is_expert"].sum()),
        }
    source_manifest = Path(cfg.dataset_manifest)
    _atomic_json(
        work_dir / "manifest.json",
        {
            "schema_version": 1,
            "kind": "validation_oof_predictions",
            "run_id": cfg.get("run_id"),
            "config": cfg.get("config"),
            "fold": fold,
            "epoch": epoch,
            "global_step": global_step,
            "prediction_weights": prediction_weights,
            "dataset_id": cfg.get("dataset_id"),
            "dataset_manifest": str(source_manifest),
            "dataset_manifest_sha256": _sha256(source_manifest),
            "label_version": cfg.get("label_version"),
            "split_version": cfg.get("split_version"),
            "class_names": metric.class_names,
            "sample_unit": sample_unit,
            "n_validation_items": count,
            "views": view_counts,
            "artifacts_sha256": artifacts,
        },
    )
    os.rename(work_dir, directory)
    return directory
