"""Portable final L185 recipe, resolved without historical config inheritance."""

import json
import os
from pathlib import Path

from skp.configs import Config

COMPONENT = Path(__file__).resolve().parents[3]
ROOT = COMPONENT.parent
cfg = Config(**json.loads((COMPONENT / "recipe.json").read_text()))
data = Path(os.environ.get("KNEE_DATA_ROOT", ROOT / "assets/data/rsna-knee-abnormality-detection"))
prepared = Path(os.environ.get("MAXVIT_METADATA", ROOT / "assets/data/maxvit"))
cfg.train_csv = str(data / "train.csv")
cfg.train_series_csv = str(data / "train_series.csv")
cfg.png_root = os.environ.get("KNEE_PNG_ROOT", str(ROOT / "assets/data/maxvit_png"))
cfg.save_dir = os.environ.get("KNEE_SAVE_DIR", str(ROOT / "outputs/maxvit"))
cfg.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
cfg.folds_csv = str(prepared / "folds.csv")
cfg.labels_jsonl = str(prepared / "labels.jsonl")
cfg.train_target_policy_json = cfg.val_target_policy_json = str(prepared / "targets.json")
cfg.dataset_manifest = str(prepared / "manifest.json")
for key in ("physical_fov_metadata_csv", "physical_fov_center_csv", "medial_lateral_position_csv"):
    setattr(cfg, key, str(ROOT / "assets/data/frozen_training_metadata.csv"))
