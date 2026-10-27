import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import argparse
from copy import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import hashlib
import importlib
import io
import json
import os
import re
import sqlite3
import tarfile
import zipfile

import cv2
import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab"))
from utils.config import setup_sys_path
setup_sys_path()
from ds_knee import physical_crop_resize, slice_plan
from utils.prepare_knee import encode_uint16

ANNOTATION_MEMBER = "Semi-Quant Scoring_ASCII/kMRI_SQ_MOAKS_BICL00.txt"
ANNOTATION_SHA256 = "15e31dec0afe8bf528786546c7b0a0359ac0d675c0b314ae8cb536aac9e80258"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def baseline_config(cfg):
    baseline = copy(cfg)
    directory = Path(cfg.oai_baseline_annotations)
    baseline.train_df = str(directory / 'train.csv')
    baseline.train_df_sha256 = None
    baseline.oai_labels = str(directory / 'labels.npz')
    baseline.oai_labels_sha256 = None
    baseline.oai_selected_archives = None
    baseline.oai_root = str(Path(cfg.oai_root) / 'baseline')
    baseline.cache_dir = cfg.oai_baseline_cache_dir
    baseline.oai_series_csv = str(Path(baseline.cache_dir) / 'series.csv')
    return baseline


def build_labels(cfg):
    identity = cfg.name
    source = Path(cfg.oai_annotations_archive)
    destination = Path(cfg.oai_labels).parent
    provenance_path = destination / "provenance.json"
    if provenance_path.exists():
        return json.loads(provenance_path.read_text())
    if sha256(source) != ANNOTATION_SHA256:
        raise ValueError("OAI annotation archive differs from the pinned release")
    train_path = Path(cfg.train_df)
    if train_path.exists():
        raise FileExistsError(train_path)
    with zipfile.ZipFile(source) as archive:
        frame = pd.read_csv(io.BytesIO(archive.read(ANNOTATION_MEMBER)), sep="|", dtype=str)
    frame["SIDE"] = frame.SIDE.str.extract(r"^(\d+)")[0].map({"1": "R", "2": "L"})
    groups = frame.groupby(["ID", "SIDE"], sort=True).indices
    keys = list(groups)
    target_blocks, masks, fields = [], [], []
    project63 = frame.READPRJ.str.startswith("63").to_numpy()
    project65 = frame.READPRJ.eq("65").to_numpy()
    for column in frame.columns:
        if not column.startswith("V00M") or column in ("V00MCMNTS", "V00MTCMNTS"):
            continue
        values = pd.to_numeric(frame[column].str.extract(r"^(-?\d+(?:\.\d+)?)(?=:|$)")[0], errors="coerce").to_numpy(float)
        variants = [(column, values.copy())]
        if column.startswith("V00MBMS"):
            variants.append((column + "_project63", np.where(project63, values, np.nan)))
            variants[0][1][project63] = np.nan
        if column.startswith(("V00MBMN", "V00MBMP", "V00MOS", "V00MGC")) or column in (
            "V00MEFFWK", "V00MSYIC", "V00MANSBUR", "V00MIPBUR", "V00MPPBUR",
            "V00MPOPCYS", "V00MITBSIG", "V00MPTSIG",
        ):
            variants[0][1][project63] = np.nan
        if column.startswith("V00MOS"):
            variants[0][1][project65] = np.nan
        for name, observed in variants:
            classes = np.unique(observed[np.isfinite(observed)])
            if len(classes) < 2:
                continue
            target = np.zeros((len(keys), len(classes)), np.float32)
            mask = np.zeros(len(keys), np.float32)
            for index, key in enumerate(keys):
                grades = observed[groups[key]]
                grades = grades[np.isfinite(grades)]
                if len(grades):
                    target[index] = (grades[:, None] == classes[None]).mean(axis=0)
                    mask[index] = 1
            fields.append({"name": name, "source_column": column, "classes": classes.tolist()})
            target_blocks.append(target)
            masks.append(mask)
    targets = np.concatenate(target_blocks, axis=1)
    label_mask = np.stack(masks, axis=1)
    keep = label_mask.sum(axis=1) > 0
    cases = pd.DataFrame(keys, columns=["patient_id", "laterality"])
    cases["case_id"] = cases.patient_id + "_" + cases.laterality
    cases["fold"] = [int(hashlib.sha256(f"oai:{patient}".encode()).hexdigest()[:8], 16) % 5
                     for patient in cases.patient_id]
    cases = cases.loc[keep].reset_index(drop=True)
    cases["target_row"] = np.arange(len(cases))
    destination.mkdir(parents=True, exist_ok=True)
    labels_path = destination / "labels.npz"
    with labels_path.open("xb") as stream:
        np.savez_compressed(stream, targets=targets[keep], mask=label_mask[keep])
    schema_path = Path(cfg.oai_schema)
    save_json(schema_path, {"fields": fields, "source_member": ANNOTATION_MEMBER,
                            "repeated_readings": "mean categorical probabilities per knee and observed field",
                            "project_masks": {"63": "separate BML-size bins; no BML counts/cystic fraction, osteophytes, effusion, Hoffa synovitis or extra-articular scores",
                                              "65": "osteophytes not assessed"}})
    save_csv(train_path, cases)
    provenance = {"identity": identity, "stage": "baseline", "source_sha256": ANNOTATION_SHA256,
                  "source_member": ANNOTATION_MEMBER, "knees": len(cases),
                  "participants": cases.patient_id.nunique(), "fields": len(fields),
                  "classes": int(targets.shape[1]), "folds": cases.fold.value_counts().sort_index().to_dict(),
                  "train_df": str(train_path), "train_df_sha256": sha256(train_path),
                  "labels_path": str(labels_path), "labels_sha256": sha256(labels_path),
                  "schema_path": str(schema_path), "schema_sha256": sha256(schema_path)}
    save_json(provenance_path, provenance)
    return provenance


def build_longitudinal_labels(cfg, manifest_path, download_selection):
    identity = cfg.name
    parent = baseline_config(cfg)
    source = Path(cfg.oai_annotations_archive)
    if sha256(source) != ANNOTATION_SHA256 or sha256(parent.oai_schema) != parent.oai_schema_sha256:
        raise ValueError("Native MOAKS source or parent task schema differs")
    destination = Path(cfg.oai_labels).parent
    train_path = Path(cfg.train_df)
    if Path(cfg.oai_labels).exists() or train_path.exists():
        raise FileExistsError(f"Native MOAKS artifacts already exist for {identity}")
    manifest = pd.read_csv(manifest_path, dtype={"OAI_ID": str})
    old = pd.read_csv(parent.train_df, dtype={"patient_id": str})
    old_series = pd.read_csv(parent.oai_series_csv, dtype={"patient_id": str})
    baseline = manifest.loc[manifest.visit_month.eq(0)]
    known = set(baseline.OAI_ID + "_" + baseline.knee_side)
    extra = []
    for row in old.loc[~old.case_id.isin(known)].itertuples(index=False):
        series = old_series.loc[old_series.case_id.eq(row.case_id)].iloc[0]
        with tarfile.open(Path(parent.oai_root) / series.archive, "r|gz") as archive:
            member = next(member for member in archive if member.isfile())
            header = pydicom.dcmread(io.BytesIO(archive.extractfile(member).read()), stop_before_pixels=True)
        if str(header.PatientID) != row.patient_id:
            raise ValueError("Native baseline participant does not match its DICOM")
        extra.append({"StudyInstanceUID": str(header.StudyInstanceUID), "OAI_ID": row.patient_id,
                      "visit_month": 0, "knee_side": row.laterality})
    augmented = pd.concat([manifest, pd.DataFrame(extra)], ignore_index=True)
    if augmented.StudyInstanceUID.duplicated().any() or augmented.duplicated(["OAI_ID", "visit_month", "knee_side"]).any():
        raise ValueError("Native MOAKS cohort identities are ambiguous")
    augmented["target_row"] = np.arange(len(augmented))
    schema = json.loads(Path(parent.oai_schema).read_text())
    fields = schema["fields"]
    targets = np.zeros((len(augmented), sum(len(field["classes"]) for field in fields)), np.float32)
    masks = np.zeros((len(augmented), len(fields)), np.float32)
    excluded, members = {}, []
    visits = {0: "00", 12: "01", 24: "03", 36: "05", 48: "06", 72: "08", 96: "10"}
    with zipfile.ZipFile(source) as archive:
        for month, cohort in augmented.groupby("visit_month", sort=True):
            visit = visits[int(month)]
            member = f"Semi-Quant Scoring_ASCII/kMRI_SQ_MOAKS_BICL{visit}.txt"
            members.append(member)
            frame = pd.read_csv(io.BytesIO(archive.read(member)), sep="|", dtype=str, encoding="latin1")
            frame.columns = frame.columns.str.upper()
            frame["knee_side"] = frame.SIDE.str.extract(r"^(\d+)")[0].map({"1": "R", "2": "L"})
            frame = cohort.merge(frame, left_on=["OAI_ID", "knee_side"], right_on=["ID", "knee_side"])
            indices = frame.target_row.to_numpy(int)
            alternate = frame.READPRJ.str.startswith("63").to_numpy() | frame.READPRJ.eq("61").to_numpy()
            project65 = frame.READPRJ.eq("65").to_numpy()
            offset = 0
            for task, field in enumerate(fields):
                canonical = field["source_column"]
                column = canonical.replace("V00", "V" + visit, 1)
                values = pd.to_numeric(frame[column].str.extract(r"^(-?\d+(?:\.\d+)?)(?=:|$)")[0],
                                       errors="coerce").to_numpy(float).copy()
                if field["name"].endswith("_project63"):
                    values[~alternate] = np.nan
                elif canonical.startswith("V00MBMS"):
                    values[alternate] = np.nan
                if canonical.startswith(("V00MBMN", "V00MBMP", "V00MOS", "V00MGC")) or canonical in (
                    "V00MEFFWK", "V00MSYIC", "V00MANSBUR", "V00MIPBUR", "V00MPPBUR",
                    "V00MPOPCYS", "V00MITBSIG", "V00MPTSIG",
                ):
                    values[alternate] = np.nan
                if canonical.startswith("V00MOS"):
                    values[project65] = np.nan
                classes = np.asarray(field["classes"])
                observed = np.isfinite(values) & np.isin(values, classes)
                unsupported = values[np.isfinite(values) & ~observed]
                if len(unsupported):
                    grades, counts = np.unique(unsupported, return_counts=True)
                    excluded[f"{month}:{field['name']}"] = {str(grade): int(count) for grade, count in zip(grades, counts)}
                counts = np.bincount(indices[observed], minlength=len(augmented))
                block = targets[:, offset:offset + len(classes)]
                np.add.at(block, indices[observed], (values[observed, None] == classes[None]).astype(np.float32))
                present = counts > 0
                block[present] /= counts[present, None]
                masks[present, task] = 1
                offset += len(classes)
    keep = masks.sum(1) > 0
    cases = augmented.loc[keep, ["StudyInstanceUID", "OAI_ID", "visit_month", "knee_side"]].rename(
        columns={"OAI_ID": "patient_id", "knee_side": "laterality"}).reset_index(drop=True)
    cases["case_id"] = cases.StudyInstanceUID
    cases["fold"] = [int(hashlib.sha256(f"oai:{patient}".encode()).hexdigest()[:8], 16) % 5
                     for patient in cases.patient_id]
    cases["target_row"] = np.arange(len(cases))
    keys = cases.loc[cases.visit_month.eq(0), "patient_id"] + "_" + cases.loc[cases.visit_month.eq(0), "laterality"]
    if set(keys) != set(old.case_id):
        raise ValueError("The native parent baseline cohort was not preserved")
    selected = pd.read_csv(download_selection, dtype={"OAI_ID": str})
    if extra:
        extra_frame = pd.DataFrame(extra)
        extra_frame["case_id"] = extra_frame.OAI_ID + "_" + extra_frame.knee_side
        additional = extra_frame.merge(old_series[["case_id", "archive"]], on="case_id", validate="one_to_many")
        metadata = pd.read_csv(parent.oai_package_metadata).rename(columns=str.lower)
        additional = additional.merge(metadata, left_on="archive", right_on="download_alias", validate="one_to_one")
        additional["archive"] = "baseline/" + additional.archive
        additional["package_id"] = parent.oai_package_id
        selected = pd.concat([selected, additional], ignore_index=True)
    columns = ["StudyInstanceUID", "OAI_ID", "visit_month", "knee_side", "archive", "package_id", "package_file_id", "file_size"]
    selected = selected[columns].drop_duplicates("archive")
    destination.mkdir(parents=True, exist_ok=True)
    target_path = destination / "labels.npz"
    with target_path.open("xb") as stream:
        np.savez_compressed(stream, targets=targets[keep], mask=masks[keep])
    selection_path = destination / "selected_archives.csv"
    save_csv(selection_path, selected)
    save_csv(destination / "study_manifest.csv", augmented.drop(columns="target_row"))
    save_json(destination / "excluded_scores.json", excluded)
    save_csv(train_path, cases)
    provenance = {"identity": identity, "baseline_train_df": parent.train_df, "source_sha256": ANNOTATION_SHA256,
                  "source_members": members, "schema_sha256": parent.oai_schema_sha256,
                  "studies": len(cases), "participants": cases.patient_id.nunique(), "fields": len(fields),
                  "classes": targets.shape[1], "preserved_baseline_knees": len(old),
                  "visits": cases.visit_month.value_counts().sort_index().to_dict(),
                  "folds": cases.fold.value_counts().sort_index().to_dict(),
                  "score_policy": "Frozen parent categories; within-grade change codes, non-assessable codes and out-of-vocabulary grades are masked without remapping",
                  "project61": "Shares documented project-63 BML bins and unassessed-feature masks",
                  "train_df": str(train_path), "train_df_sha256": sha256(train_path),
                  "labels_path": str(target_path), "labels_sha256": sha256(target_path),
                  "selected_archives": str(selection_path), "selected_archives_sha256": sha256(selection_path),
                  "source_manifest_sha256": sha256(manifest_path)}
    save_json(destination / "provenance.json", provenance)
    return provenance


def header_record(task):
    alias, root = task
    path = Path(root) / alias
    with tarfile.open(path, "r|gz") as archive:
        member = next(member for member in archive if member.isfile())
        header = pydicom.dcmread(io.BytesIO(archive.extractfile(member).read()), stop_before_pixels=True)
    description = str(getattr(header, "SeriesDescription", "")).upper()
    tokens = re.split(r"[_\W]+", description)
    side = next((side for name, side in [("RIGHT", "R"), ("LEFT", "L")] if name in tokens), None)
    side = side or str(getattr(header, "ImageLaterality", "") or getattr(header, "Laterality", ""))
    selected = (str(getattr(header, "Modality", "")) == "MR"
                and str(getattr(header, "BodyPartExamined", "")).upper().strip() == "KNEE"
                and side in ("R", "L")
                and any(word in description for word in ("IW_TSE", "DESS", "MPR", "FLASH")))
    patient = str(header.PatientID).strip()
    record = {"archive": alias, "patient_id": patient, "laterality": side,
              "case_id": f"{patient}_{side}", "description": description, "selected": selected}
    if selected:
        orientation = np.asarray(header.ImageOrientationPatient, float)
        normal = np.cross(orientation[:3], orientation[3:])
        plane = ("SAG", "COR", "AXI")[int(np.abs(normal).argmax())]
        record.update(series_uid=str(header.SeriesInstanceUID), plane=plane,
                      spacing_mm=float(np.mean(np.asarray(header.PixelSpacing, float))))
    return record


def completed_archives(cfg, patients):
    directory = Path.home() / "NDA/nda-tools/downloadcmd/packages" / str(cfg.oai_package_id) / ".download-progress"
    aliases = set()
    for report in directory.glob("*/download-progress-report.csv"):
        with report.open(newline="") as stream:
            for row in csv.DictReader(stream):
                alias = row.get("package_file_expected_location", "")
                parts = Path(alias).parts
                if (row.get("exists") == "True" and alias.endswith(".tar.gz") and len(parts) >= 6
                        and parts[3] in patients and row.get("actual_file_size") == row.get("expected_file_size")):
                    aliases.add(alias)
    return sorted(aliases)


def index_archives(cfg):
    cases = pd.read_csv(cfg.train_df, dtype={"patient_id": str})
    destination = Path(cfg.cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(destination / "index.sqlite")
    database.execute("CREATE TABLE IF NOT EXISTS series (archive TEXT PRIMARY KEY, record TEXT NOT NULL)")
    indexed = {row[0] for row in database.execute("SELECT archive FROM series")}
    pending = sorted(set(completed_archives(cfg, set(cases.patient_id))) - indexed)
    if cfg.oai_index_limit:
        pending = pending[:cfg.oai_index_limit]
    print(f"OAI index: {len(pending)} completed archives to inspect", flush=True)
    with ThreadPoolExecutor(max_workers=cfg.cache_workers) as pool:
        for count, record in enumerate(pool.map(header_record, [(alias, cfg.oai_root) for alias in pending]), 1):
            database.execute("INSERT INTO series VALUES (?, ?)", (record["archive"], json.dumps(record)))
            if count % 100 == 0:
                database.commit()
            if count % 1000 == 0:
                print(f"OAI index: {count}/{len(pending)}", flush=True)
    database.commit()
    records = [json.loads(row[0]) for row in database.execute("SELECT record FROM series ORDER BY archive")]
    database.close()
    case_ids = set(cases.case_id)
    selected = pd.DataFrame([row for row in records if row["selected"] and row["case_id"] in case_ids])
    if not selected.empty:
        selected = selected.drop_duplicates(["case_id", "series_uid"])
    save_csv(destination / "indexed_series.csv", selected)
    print(f"OAI index: {len(selected)} usable MRI series", flush=True)
    return selected


def cache_series(task):
    record, values = task
    from types import SimpleNamespace
    cfg = SimpleNamespace(**values)
    path = Path(cfg.oai_root) / record["archive"]
    key = hashlib.sha256(("cfg_1_77_float32_geometry_v2" + record["case_id"] + record["series_uid"]).encode()).hexdigest()[:24]
    destination = Path(cfg.cache_dir) / "series" / key
    if destination.with_suffix(".json").exists():
        return json.loads(destination.with_suffix(".json").read_text())
    slice_groups = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            header = pydicom.dcmread(archive.extractfile(member))
            orientation = np.asarray(header.ImageOrientationPatient, float)
            coordinate = float(np.asarray(header.ImagePositionPatient, float) @ np.cross(orientation[:3], orientation[3:]))
            image = header.pixel_array.astype(np.float32)
            image *= float(getattr(header, "RescaleSlope", 1) or 1)
            image += float(getattr(header, "RescaleIntercept", 0) or 0)
            if str(header.PhotometricInterpretation) == "MONOCHROME1":
                image = image.max() - image
            geometry = (tuple(orientation), tuple(float(value) for value in header.PixelSpacing), image.shape)
            slice_groups.setdefault(geometry, []).append((coordinate, int(header.InstanceNumber), image))
    # OAI MPR archives can include an orthogonal reference frame in the same series.
    geometry, slices = max(slice_groups.items(), key=lambda item: len(item[1]))
    orientation, spacing, _ = geometry
    normal = np.cross(orientation[:3], orientation[3:])
    record = dict(record, plane=("SAG", "COR", "AXI")[int(np.abs(normal).argmax())],
                  spacing_mm=float(np.mean(spacing)))
    slices.sort(key=lambda row: (row[0], row[1]))
    volume, _, _ = encode_uint16(np.stack([row[2] for row in slices]))
    if record["laterality"] == "R":
        volume = volume[::-1] if record["plane"] == "SAG" else volume[:, :, ::-1]
    indices, real = slice_plan(len(volume), cfg.cache_slices)
    sampled = volume[indices].astype(np.float32)
    # Match the parent's float32 arithmetic across NumPy promotion modes.
    low, high = (float(value) for value in np.percentile(sampled[real > 0], [cfg.pct_low, cfg.pct_high]))
    if high <= low:
        high = low + 1
    images = np.stack([physical_crop_resize(cfg, np.clip((image - low) / (high - low), 0, 1) * 255,
                                          record["spacing_mm"]).astype(np.uint8) for image in sampled])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, images=images, smask=real)
    temporary.replace(destination.with_suffix(".npz"))
    result = {**record, "cache_path": str(destination.with_suffix(".npz")), "n_slices": len(volume)}
    save_json(destination.with_suffix(".json"), result)
    return result


def prepare_cache(cfg):
    records = index_archives(cfg)
    if records.empty:
        raise ValueError("No completed annotated knee MRI series are available")
    if cfg.oai_cache_limit:
        records = records.iloc[:cfg.oai_cache_limit]
    values = {key: getattr(cfg, key) for key in ("oai_root", "cache_dir", "cache_slices", "img_px",
                                                "crop_mm", "pct_low", "pct_high")}
    tasks = [(record, values) for record in records.to_dict("records")]
    cached = []
    with ProcessPoolExecutor(max_workers=cfg.cache_workers, initializer=cv2.setNumThreads, initargs=(0,)) as pool:
        for count, record in enumerate(pool.map(cache_series, tasks, chunksize=1), 1):
            cached.append(record)
            if count % 100 == 0 or count == len(tasks):
                print(f"OAI cache: {count}/{len(tasks)}", flush=True)
    output = pd.DataFrame(cached)
    save_csv(cfg.oai_series_csv, output)
    print(f"OAI cache ready: {len(output)} series, {output.case_id.nunique()} knees", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_18_7")
    parser.add_argument("--baseline", action="store_true", help="Prepare baseline image cache before longitudinal labels")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--longitudinal", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--selection")
    parser.add_argument("--index-limit", type=int, default=0)
    parser.add_argument("--cache-limit", type=int, default=0)
    args = parser.parse_args()
    cfg = copy(importlib.import_module(args.config).cfg)
    if args.longitudinal:
        print(json.dumps(build_longitudinal_labels(cfg, args.manifest, args.selection), indent=2))
        return
    if args.labels:
        print(json.dumps(build_labels(baseline_config(cfg)), indent=2))
        return
    if args.baseline:
        cfg = baseline_config(cfg)
    from utils.artifacts import resolve_artifacts
    resolve_artifacts(cfg)
    cfg.oai_index_limit = args.index_limit
    cfg.oai_cache_limit = args.cache_limit
    if args.baseline:
        prepare_cache(cfg)
    else:
        importlib.import_module(cfg.oai_cache_prepare_module).prepare_cache(cfg)


if __name__ == "__main__":
    main()
