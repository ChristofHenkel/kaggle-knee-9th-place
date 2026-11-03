"""Prepare, check, train and export the original L185 dual-view FS router."""

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "maxvit"
UID = "StudyInstanceUID"
LABEL_HASH = "e9018032ee2eab75df52ff3a69dbf413556b3c51094146542deb1905597af692"
GEOMETRY_HASH = "42678282f291af3c6b4793db1e019bd1a340c0d709f5df061336e1ee06b42062"


def setup():
    sys.path[:0] = [str(COMPONENT / "src"), str(COMPONENT / "scripts")]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config():
    setup()
    return copy.deepcopy(importlib.import_module("skp.configs.l185_public").cfg)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def effective_targets(labels, official, names):
    """Use supplied soft values except for the 58 official binary annotations."""
    import numpy as np

    if not labels.index.is_unique or not official.index.is_unique or set(labels.index) != set(official.index):
        raise ValueError("Label and official study identifiers must have exact unique coverage")
    targets = labels[names].copy()
    if not np.isfinite(targets.values).all() or not ((targets.values >= 0) & (targets.values <= 1)).all():
        raise ValueError("Invalid label probabilities")
    truth = official.loc[labels.index, names]
    partial = truth.notna().any(axis=1) & ~truth.notna().all(axis=1)
    if partial.any():
        raise ValueError("Partially populated gold annotations")
    gold = truth.notna().all(axis=1)
    if not np.isin(truth.loc[gold].values, [0, 1]).all():
        raise ValueError("Gold annotations must be binary")
    targets.loc[gold] = truth.loc[gold]
    return targets, gold


def prepare(args):
    import pandas as pd

    cfg = config()
    directory = Path(cfg.dataset_manifest).parent
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite prepared metadata: {directory}")
    label_path = ROOT / "assets/data/labels_1_85.csv"
    if sha(label_path) != LABEL_HASH:
        raise ValueError("Bundled L185 labels have changed")
    if sha(ROOT / "assets/data/frozen_training_metadata.csv") != GEOMETRY_HASH:
        raise ValueError("Bundled geometry has changed")
    labels = pd.read_csv(label_path, dtype={UID: str}).set_index(UID)
    official = pd.read_csv(cfg.train_csv, dtype={UID: str}).set_index(UID)
    targets, gold = effective_targets(labels, official, cfg.class_names)
    if len(labels) != 4407 or int(gold.sum()) != 58:
        raise ValueError("Expected 4407 studies and 58 gold studies")
    if not (gold.astype(int) == labels.is_gold.astype(int)).all():
        raise ValueError("Bundled gold membership differs from official truth")
    if labels.fold.value_counts().sort_index().to_dict() != {0: 882, 1: 882, 2: 881, 3: 881, 4: 881}:
        raise ValueError("Unexpected fold distribution")
    if (labels.groupby("fold_group").fold.nunique() != 1).any():
        raise ValueError("A report group crosses folds")
    directory.mkdir(parents=True)
    folds = labels[["fold"]].rename(columns={"fold": "cv_fold"})
    folds["always_validation"] = 0
    folds["expert_labeled"] = gold.astype(int)
    folds.to_csv(cfg.folds_csv)
    with Path(cfg.labels_jsonl).open("x") as stream:
        for uid, row in labels.iterrows():
            stream.write(json.dumps({UID: uid, "labels": {
                name: {"value": float(row[name])} for name in cfg.class_names}}) + "\n")
    write_json(Path(cfg.train_target_policy_json), {
        "class_names": cfg.class_names,
        "study_targets": {uid: row.tolist() for uid, row in targets.iterrows()},
        "study_weights": {uid: [1.0] * 12 for uid in targets.index},
        "label_version_id": cfg.label_version,
    })
    paths = [label_path, Path(cfg.train_csv), Path(cfg.train_series_csv),
             ROOT / "assets/data/frozen_training_metadata.csv", Path(cfg.folds_csv),
             Path(cfg.labels_jsonl), Path(cfg.train_target_policy_json)]
    write_json(Path(cfg.dataset_manifest), {
        "dataset_id": cfg.dataset_id, "label_version": cfg.label_version,
        "split_version": cfg.split_version,
        "components": [{"path": str(p.resolve()), "sha256": sha(p)} for p in paths],
        "gold_policy": "official binary targets in train and validation; rotating gold folds",
        "confidence_weighting": False, "oof_fusion": False,
    })
    print(f"Prepared {len(targets)} studies; {gold.sum()} official gold overrides: {directory}")


def verify_manifest(cfg):
    for item in json.loads(Path(cfg.dataset_manifest).read_text())["components"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"Prepared input changed: {item['path']}")


def check(args):
    import pandas as pd
    import torch
    from skp.datasets.rsna_knee_png import Dataset
    from skp.optim import get_scheduler

    cfg = config()
    cfg.fold = args.fold
    verify_manifest(cfg)
    train, val = Dataset(cfg, "train"), Dataset(cfg, "val")
    # Do not accept a partial conversion just because every study has one series.
    keys = [UID, "SeriesInstanceUID"]
    series = pd.read_csv(cfg.train_series_csv)
    geometry = pd.read_csv(cfg.physical_fov_center_csv, low_memory=False)
    expected = series[series.Fluid_Sensitive == 1].merge(geometry, on=keys, validate="one_to_one")
    expected = expected[(expected.coordinate_status == "ok") & expected.position_axis.notna()]
    expected_ids = set(expected[keys].itertuples(index=False, name=None))
    actual_ids = {(r["study_uid"], r["series_uid"]) for r in train.items + val.items}
    if actual_ids != expected_ids or len(actual_ids) != 14008:
        raise ValueError("PNG coverage differs from the 14008 eligible FS series; complete conversion first")
    train_uids = {r["study_uid"] for r in train.items}
    val_uids = {r["study_uid"] for r in val.items}
    if train_uids & val_uids or len(train_uids | val_uids) != 4407:
        raise ValueError("Missing study coverage or train/validation leakage")
    cfg.world_size = 1
    cfg.args = {"strategy": "ddp"}
    schedule = get_scheduler(cfg, torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=2e-4), train_dataset=train)
    if (schedule.total_steps, schedule.warmup_steps) != (4400, 440):
        raise ValueError("Expected corrected 4400 total / 440 warmup optimizer updates")
    for plane in range(3):
        index = next(i for i, row in enumerate(train.items) if row["anatomical_plane_index"] == plane)
        sample = train[index]
        if tuple(sample["x"].shape) != (32, 3, 256, 256):
            raise ValueError("Unexpected series tensor geometry")
    print(f"fold={args.fold}: train={len(train_uids)} studies/{len(train)} series; "
          f"val={len(val_uids)} studies/{len(val)} series; schedule=4400/440; all planes checked")
    return cfg


def download(args):
    from huggingface_hub import hf_hub_download

    cfg = config()
    cache = ROOT / "assets/weights/maxvit-hf/hub"
    file = hf_hub_download(repo_id="timm/" + cfg.backbone, filename="model.safetensors",
                           revision=cfg.pretrained_revision, cache_dir=cache)
    if sha(file) != cfg.pretrained_sha256:
        raise ValueError("ImageNet initialization checksum mismatch")
    ref = cache / ("models--timm--" + cfg.backbone) / "refs/main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(cfg.pretrained_revision)
    print(file)


def convert(args):
    cfg = config()
    subprocess.run([sys.executable, str(COMPONENT / "scripts/convert_dicom_to_png.py"),
                    "--data-root", str(Path(cfg.train_csv).parent), "--output-root", cfg.png_root,
                    "--splits", "train", "--workers", str(args.workers), "--skip-malformed-series"],
                   env=child_env(), check=True)


def child_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(COMPONENT / "src"), str(COMPONENT / "scripts")])
    # Original workstation-specific NCCL workaround must not alter other hosts.
    env["GCP_NCCL_SOCKET_HACK_DISABLE"] = "1"
    return env


def train(args):
    import torch
    from skp.tracking import validate_mlflow_tracking

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run-id must contain only letters, digits, underscores and hyphens")
    cfg = check(args)
    validate_mlflow_tracking(cfg.mlflow_tracking_uri, allow_local=False, save_dir=cfg.save_dir)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    cache = ROOT / "assets/weights/maxvit-hf"
    weight = cache / "hub" / ("models--timm--" + cfg.backbone) / "snapshots" / cfg.pretrained_revision / "model.safetensors"
    if sha(weight) != cfg.pretrained_sha256:
        raise ValueError("Run the download command to obtain pinned ImageNet weights")
    ref = cache / "hub" / ("models--timm--" + cfg.backbone) / "refs/main"
    if ref.read_text().strip() != cfg.pretrained_revision:
        raise ValueError("Pinned ImageNet cache reference changed; rerun download")
    env = child_env()
    env.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"),
               HUGGINGFACE_HUB_CACHE=str(cache / "hub"), HF_HUB_OFFLINE="1")
    command = [sys.executable, str(COMPONENT / "scripts/train_two_view_ema.py"), "l185_public",
               "--run_id", args.run_id, "--kfold", str(args.fold), "--devices", "1",
               "--strategy", "ddp", "--precision", "bf16-mixed"]
    log_root = Path(cfg.save_dir) / "launches"
    log_root.mkdir(parents=True, exist_ok=True)
    record = log_root / f"{args.run_id}-fold{args.fold}.json"
    write_json(record, {"command": command, "dataset_manifest_sha256": sha(cfg.dataset_manifest),
                       "recipe_sha256": sha(COMPONENT / "recipe.json"), "status": "running"})
    result = subprocess.run(command, env=env, cwd=ROOT)
    state = json.loads(record.read_text())
    state.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
    record.write_text(json.dumps(state, indent=2) + "\n")
    raise SystemExit(result.returncode)


def export(args):
    import torch

    cfg = config()
    cfg.pretrained = False
    cfg.enable_gradient_checkpointing = False
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("epoch") != 19 or payload.get("prediction_weights") != "ema"
            or payload.get("global_step") != 4400):
        raise ValueError("Export requires final epoch-20 EMA checkpoint, not best or raw")
    state = {k.removeprefix("model."): v for k, v in payload["state_dict"].items() if k.startswith("model.")}
    model = importlib.import_module("skp.models." + cfg.model).Net(cfg)
    model.load_state_dict(state, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state}, args.output)
    write_json(args.output.with_suffix(".json"), {"sha256": sha(args.output), "source_sha256": sha(args.checkpoint),
                                                "epoch": 19, "prediction_weights": "ema", "tensors": len(state)})
    print(args.output)


def pool(args):
    """Pool final held-out study probabilities, never an in-sample CV5 ensemble."""
    import numpy as np
    import pandas as pd
    from skp.metrics.rsna_knee import _metric_name

    cfg = config()
    verify_manifest(cfg)
    if args.output.exists():
        raise FileExistsError(args.output)
    folds = pd.read_csv(cfg.folds_csv).set_index(UID)
    frames, seen = [], set()
    for directory in args.oof_dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        fold = manifest["fold"]
        if (fold not in range(5) or fold in seen or manifest["epoch"] != 19
                or manifest["prediction_weights"] != "ema" or manifest["global_step"] != 4400
                or manifest["class_names"] != cfg.class_names
                or manifest["dataset_manifest_sha256"] != sha(cfg.dataset_manifest)):
            raise ValueError(f"Wrong fold/epoch/weights/data in {directory}")
        path = directory / "studies_all_series.csv"
        if sha(path) != manifest["artifacts_sha256"][path.name]:
            raise ValueError(f"OOF checksum mismatch: {path}")
        frame = pd.read_csv(path)
        expected = set(folds.index[folds.cv_fold == fold])
        if not frame[UID].is_unique or set(frame[UID]) != expected:
            raise ValueError(f"OOF coverage is not exactly held-out fold {fold}")
        frame["fold"] = fold
        seen.add(fold)
        frames.append(frame)
    if seen != set(range(5)):
        raise ValueError("Exactly five different folds required")
    result = pd.concat(frames, ignore_index=True).sort_values(UID)
    pred_columns = ["pred_" + _metric_name(name) for name in cfg.class_names]
    pred = result[pred_columns].to_numpy()
    if not np.isfinite(pred).all() or not ((pred >= 0) & (pred <= 1)).all():
        raise ValueError("Invalid OOF probabilities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    write_json(args.output.with_suffix(".json"), {"sha256": sha(args.output),
        "folds": sorted(seen), "prediction_weights": "ema", "epoch": 19,
        "oof_dirs": [str(p) for p in args.oof_dirs], "dataset_manifest_sha256": sha(cfg.dataset_manifest)})
    print(f"Saved {len(result)} held-out study predictions to {args.output}")


def main():
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("download")
    commands.add_parser("convert").add_argument("--workers", type=int, default=8)
    for name in ("check", "train"):
        p = commands.add_parser(name)
        p.add_argument("--fold", type=int, choices=range(5), required=True)
        if name == "train":
            p.add_argument("--run-id", required=True)
    p = commands.add_parser("pool")
    p.add_argument("--oof-dirs", type=Path, nargs=5, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("export")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
