from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from .artifacts import sha256
from .config import setup_sys_path
from .prepare_knee_labels import (
    SPLIT, TARGETS, canonical, save, training_table)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "assets/data"
UID = "StudyInstanceUID"
EXPECTED_FOLD_ROWS = {0: 882, 1: 882, 2: 881, 3: 881, 4: 881}
CV_EXPERIMENTS = ("v307", "v310")
FULLFIT_EXPERIMENTS = ("v311", "v313")
BASE_LABELS = DATA / "labels_1_85.csv"
BASE_LABELS_SHA256 = (
    "e9018032ee2eab75df52ff3a69dbf413556b3c51094146542deb1905597af692")


@dataclass(frozen=True)
class Experiment:
    config: str
    kind: str
    seed: int
    epochs: int
    source_labels: Path | None = None
    oof_source: str | None = None
    recipe: str | None = None


EXPERIMENTS = {
    "v307": Experiment(
        config="cfg_v307", kind="cv", seed=2026, epochs=14),
    "v310": Experiment(
        config="cfg_v310", kind="cv", seed=2026, epochs=12),
    "v311": Experiment(
        config="cfg_v311", kind="fullfit", seed=3026, epochs=19,
        source_labels=BASE_LABELS, oof_source="v310",
        recipe=(
            "non-gold: 0.5 * L85 + "
            "0.5 * v310 final epoch-12 EMA OOF; "
            "gold: L85 unchanged")),
    "v313": Experiment(
        config="cfg_v313", kind="fullfit", seed=4026, epochs=21,
        source_labels=BASE_LABELS, oof_source="v307",
        recipe=(
            "non-gold: 0.5 * L85 + "
            "0.5 * v307 final epoch-14 EMA OOF; "
            "gold: L85 unchanged")),
}


def read_table(path):
    frame = pd.read_csv(
        path, dtype={UID: str}, float_precision="round_trip")
    frame.columns = frame.columns.str.strip()
    frame[UID] = frame[UID].str.strip()
    frame = frame.set_index(UID)
    if frame.index.has_duplicates:
        raise ValueError(f"{path}: duplicate study identifiers")
    return frame


def portable(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def validate_probability_table(frame, name):
    if len(frame) != 4407 or frame.index.has_duplicates:
        raise ValueError(f"{name}: expected 4,407 unique study identifiers")
    missing = [target for target in TARGETS if target not in frame]
    if missing:
        raise ValueError(f"{name}: missing targets {missing}")
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name}: target values must be finite probabilities")


def validate_canonical(base):
    validate_probability_table(base, "canonical labels")
    counts = base.fold.value_counts().sort_index().to_dict()
    if counts != EXPECTED_FOLD_ROWS:
        raise ValueError(
            f"Canonical fold sizes differ: {counts} != {EXPECTED_FOLD_ROWS}")
    if int(base.is_gold.sum()) != 58:
        raise ValueError("Canonical labels must contain 58 gold studies")


def fuse_tables(source, oof, base):
    validate_canonical(base)
    validate_probability_table(source, "source labels")
    validate_probability_table(oof, "OOF predictions")
    if set(source.index) != set(base.index) or set(oof.index) != set(base.index):
        raise ValueError("Source, OOF and canonical study sets differ")
    source = source.loc[base.index]
    oof = oof.loc[base.index]
    if "fold" not in oof:
        raise ValueError("OOF predictions must include fold")
    if not np.array_equal(
            oof.fold.to_numpy(np.int8), base.fold.to_numpy(np.int8)):
        raise ValueError("OOF folds differ from canonical folds")
    if "is_gold" in oof and not np.array_equal(
            oof.is_gold.to_numpy(bool), base.is_gold.to_numpy(bool)):
        raise ValueError("OOF gold flags differ from canonical labels")

    result = source[TARGETS].copy()
    non_gold = ~base.is_gold.to_numpy(bool)
    result.loc[non_gold, TARGETS] = (
        0.5 * source.loc[non_gold, TARGETS].to_numpy(np.float64)
        + 0.5 * oof.loc[non_gold, TARGETS].to_numpy(np.float64))
    result[SPLIT] = base[SPLIT]
    result.index.name = UID
    return result


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(
            temporary, index=True, lineterminator="\n",
            float_format="%.17g")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_oof_provenance(experiment, oof_path):
    spec = EXPERIMENTS[experiment]
    manifest_path = oof_path.with_suffix(".provenance.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"OOF provenance is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "experiment": experiment,
        "config": spec.config,
        "epoch": spec.epochs,
        "variant": "ema",
        "rows": 4407,
        "output_sha256": sha256(oof_path),
        "source_labels_sha256": BASE_LABELS_SHA256,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{experiment} OOF provenance mismatch: {mismatches}")
    return manifest_path


def write_fusion(experiment, source_path, oof_path, output_path, force):
    spec = EXPERIMENTS[experiment]
    source_path, oof_path = Path(source_path), Path(oof_path)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} exists; pass --force to reproduce it")
    expected_source_hash = BASE_LABELS_SHA256
    actual_source_hash = sha256(source_path)
    if actual_source_hash != expected_source_hash:
        raise ValueError(
            f"Source-label checksum mismatch: {actual_source_hash} "
            f"!= {expected_source_hash}")
    provenance_path = validate_oof_provenance(spec.oof_source, oof_path)

    base = canonical()
    result = fuse_tables(
        read_table(source_path), read_table(oof_path), base)
    atomic_csv(result, output_path)
    provenance = {
        "experiment": experiment,
        "recipe": spec.recipe,
        "sources": {
            "source_labels": {
                "file": portable(source_path),
                "sha256": actual_source_hash,
            },
            "oof_predictions": {
                "file": portable(oof_path),
                "sha256": sha256(oof_path),
                "provenance": portable(provenance_path),
                "provenance_sha256": sha256(provenance_path),
            },
            "canonical_labels": {
                "file": "assets/data/labels_1_85.csv",
                "sha256": sha256(DATA / "labels_1_85.csv"),
            },
        },
        "output": portable(output_path),
        "output_sha256": sha256(output_path),
        "rows": len(result),
        "gold_rows": int(base.is_gold.sum()),
    }
    atomic_json(
        provenance, output_path.with_suffix(".provenance.json"))
    return result


def prepare(args):
    spec = EXPERIMENTS[args.experiment]
    competition = Path(args.competition)
    if spec.kind == "cv":
        if args.source_labels or args.oof or args.output_labels:
            raise ValueError(
                "v307/v310 use the bundled labels_1_85.csv")
        if sha256(BASE_LABELS) != BASE_LABELS_SHA256:
            raise ValueError("Bundled L85 labels changed")
        labels = read_table(BASE_LABELS)
        label_path = BASE_LABELS
    else:
        source_path = Path(args.source_labels or spec.source_labels)
        oof_path = Path(
            args.oof
            or ROOT / "outputs" / EXPERIMENTS[spec.oof_source].config
            / "final_oof.csv")
        label_path = Path(
            args.output_labels or DATA / f"labels_{args.experiment}.csv")
        labels = write_fusion(
            args.experiment, source_path, oof_path, label_path, args.force)

    validate_canonical(canonical())
    validate_probability_table(labels, str(label_path))
    if not all(field in labels for field in SPLIT):
        raise ValueError(f"{label_path}: missing frozen split metadata")

    base_train = DATA / "train_1_85.csv"
    if not base_train.exists():
        save(
            training_table(canonical(), competition), base_train,
            [DATA / "labels_1_85.csv", competition / "train.csv"],
            "L85 with exact gold targets")
    training_output = Path(
        args.training_output or DATA / f"train_{args.experiment}.csv")
    save(
        training_table(labels, competition), training_output,
        [label_path, competition / "train.csv"],
        f"{args.experiment} effective targets with exact gold and frozen folds")
    return training_output


def collect_oof(args):
    experiment = args.experiment
    spec = EXPERIMENTS[experiment]
    setup_sys_path()
    cfg = importlib.import_module(spec.config).cfg
    run_root = Path(
        args.run_root or ROOT / "outputs" / spec.config)
    output_path = Path(args.output or run_root / "final_oof.csv")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"{output_path} exists; pass --force to replace it")

    train_path = Path(cfg.train_df)
    frame = pd.read_csv(
        train_path, dtype={UID: str}, float_precision="round_trip")
    if len(frame) != 4407 or frame[UID].duplicated().any():
        raise ValueError(
            f"{train_path}: expected 4,407 unique studies")
    if frame.fold.value_counts().sort_index().to_dict() != EXPECTED_FOLD_ROWS:
        raise ValueError("Training-table folds differ from canonical folds")

    predictions = pd.DataFrame(
        index=frame[UID], columns=TARGETS, dtype=np.float64)
    fold_sources = []
    for fold, expected_rows in EXPECTED_FOLD_ROWS.items():
        folder = run_root / f"fold{fold}"
        prediction_files = list(
            folder.glob("val_ema_data_seed*.pth"))
        if len(prediction_files) != 1:
            raise ValueError(
                f"Expected one final EMA prediction tensor in {folder}; "
                f"found {len(prediction_files)}")
        prediction_path = prediction_files[0]
        seed = prediction_path.stem.split("seed")[-1]
        expected_seed = spec.seed + fold
        if int(seed) != expected_seed:
            raise ValueError(
                f"Fold {fold} seed mismatch: {seed} != {expected_seed}")
        history_path = folder / f"history_seed{seed}.csv"
        checkpoint_path = folder / f"checkpoint_last_seed{seed}.pth"
        if not history_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold} lacks final history/checkpoint for seed {seed}")
        history = pd.read_csv(history_path)
        if int(history.epoch.iloc[-1]) != spec.epochs:
            raise ValueError(
                f"Fold {fold} history did not reach epoch {spec.epochs}")
        with torch.serialization.safe_globals([defaultdict, list]):
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu",
                weights_only=True, mmap=True)
        if int(checkpoint["epoch"]) + 1 != spec.epochs:
            raise ValueError(
                f"Fold {fold} checkpoint did not reach epoch {spec.epochs}")
        if "ema" not in checkpoint:
            raise ValueError(
                f"Fold {fold} final checkpoint has no EMA state")
        saved = torch.load(
            prediction_path, map_location="cpu", weights_only=True)
        fold_predictions = saved["predictions"].cpu().numpy()
        held_out = frame.fold.eq(fold)
        fold_uids = frame.loc[held_out, UID].tolist()
        if len(fold_uids) != expected_rows:
            raise ValueError(
                f"Fold {fold} has {len(fold_uids)} rows, "
                f"expected {expected_rows}")
        if fold_predictions.shape != (expected_rows, len(TARGETS)):
            raise ValueError(
                f"Fold {fold} prediction shape mismatch: "
                f"{fold_predictions.shape}")
        predictions.loc[fold_uids, TARGETS] = fold_predictions
        fold_sources.append({
            "fold": fold,
            "seed": int(seed),
            "rows": expected_rows,
            "predictions": portable(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "history": portable(history_path),
            "history_sha256": sha256(history_path),
            "checkpoint": portable(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
        })

    values = predictions[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Pooled OOF predictions are invalid")
    output = predictions.loc[frame[UID], TARGETS].copy()
    output.insert(0, "is_gold", frame.is_gold.to_numpy(np.int8))
    output.insert(0, "fold", frame.fold.to_numpy(np.int8))
    output.index.name = UID
    atomic_csv(output, output_path)
    provenance = {
        "experiment": experiment,
        "config": spec.config,
        "epoch": spec.epochs,
        "variant": "ema",
        "rows": len(output),
        "source_labels": portable(BASE_LABELS),
        "source_labels_sha256": sha256(BASE_LABELS),
        "training_table": portable(train_path),
        "training_table_sha256": sha256(train_path),
        "folds": fold_sources,
        "output": portable(output_path),
        "output_sha256": sha256(output_path),
    }
    atomic_json(
        provenance, output_path.with_suffix(".provenance.json"))
    print(output_path)
    return output_path


def command_for_train(args):
    spec = EXPERIMENTS[args.experiment]
    if spec.kind == "cv":
        output = Path(
            args.output_dir or ROOT / "outputs" / spec.config)
        fold = args.fold
        seed_args = []
    else:
        output = Path(
            args.output_dir
            or ROOT / "outputs" / spec.config / f"repeat{args.repeat}")
        fold = -1
        seed_args = ["--seed", str(spec.seed + args.repeat - 1)]
    command = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
    ]
    if args.master_port is not None:
        command.append(f"--master_port={args.master_port}")
    command += [
        "lab/main.py", "-C", spec.config,
        "--fold", str(fold), *seed_args,
        "--output_dir", str(output),
    ]
    if args.checkpoint:
        main_index = command.index("lab/main.py")
        command[main_index + 1:main_index + 1] = [
            "--checkpoint", args.checkpoint]
    return command


def command_for_cache(args):
    command = [
        sys.executable, "scripts/dicom_series_to_processed_npz.py",
        "--input-root", str(args.input_root),
        "--output-dir", str(ROOT / "assets/data/cached_shards"),
        "--workers", str(args.workers),
        "--laterality-workers", str(args.laterality_workers),
    ]
    if args.study_limit is not None:
        command += ["--study-limit", str(args.study_limit)]
    return command


def execute(command, dry_run):
    print(shlex.join(str(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and train the v307/v310/v311/v313 models",
        allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--experiment", choices=EXPERIMENTS, required=True)
    prepare_parser.add_argument("--source-labels")
    prepare_parser.add_argument("--oof")
    prepare_parser.add_argument("--output-labels")
    prepare_parser.add_argument("--training-output")
    prepare_parser.add_argument(
        "--competition",
        default=DATA / "rsna-knee-abnormality-detection")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(function=prepare)

    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument(
        "--experiment", choices=EXPERIMENTS, required=True)
    cache_parser.add_argument("--workers", type=int, default=16)
    cache_parser.add_argument("--laterality-workers", type=int, default=8)
    cache_parser.add_argument(
        "--input-root",
        default=DATA / "rsna-knee-abnormality-detection")
    cache_parser.add_argument("--study-limit", type=int)
    cache_parser.add_argument("--dry-run", action="store_true")
    cache_parser.set_defaults(
        function=lambda args: execute(command_for_cache(args), args.dry_run))

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--experiment", choices=EXPERIMENTS, required=True)
    train_parser.add_argument("--fold", type=int, choices=range(5))
    train_parser.add_argument("--repeat", type=int, choices=range(1, 6))
    train_parser.add_argument("--nproc", type=int, default=1)
    train_parser.add_argument("--master-port", type=int)
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--checkpoint")
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.set_defaults(
        function=lambda args: execute(command_for_train(args), args.dry_run))

    oof_parser = subparsers.add_parser("oof")
    oof_parser.add_argument(
        "--experiment", choices=CV_EXPERIMENTS, required=True)
    oof_parser.add_argument("--run-root")
    oof_parser.add_argument("--output")
    oof_parser.add_argument("--force", action="store_true")
    oof_parser.set_defaults(function=collect_oof)

    args = parser.parse_args()
    if getattr(args, "nproc", 1) < 1:
        parser.error("--nproc must be positive")
    if args.command == "train":
        kind = EXPERIMENTS[args.experiment].kind
        if kind == "cv" and (args.fold is None or args.repeat is not None):
            parser.error("v307/v310 training requires --fold and no --repeat")
        if kind == "fullfit" and (
                args.repeat is None or args.fold is not None):
            parser.error("v311/v313 training requires --repeat and no --fold")
    args.function(args)


if __name__ == "__main__":
    main()
