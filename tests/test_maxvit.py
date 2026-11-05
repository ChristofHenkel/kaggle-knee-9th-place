"""CPU contracts for the isolated L185 MaxViT component; no data downloads."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from maxvit.cli import ROOT, config, effective_targets, setup, LABEL_HASH, GEOMETRY_HASH

setup()


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None
        for name in ("lightning", "torchmetrics", "mlflow", "petname", "dotenv")),
    "Install maxvit/requirements.txt to run the isolated MaxViT tests",
)
class MaxViTTests(unittest.TestCase):
    def test_shared_artifacts_and_recipe(self):
        for file, digest in [("labels_1_85.csv", LABEL_HASH), ("frozen_training_metadata.csv", GEOMETRY_HASH)]:
            self.assertEqual(hashlib.sha256((ROOT / "assets/data" / file).read_bytes()).hexdigest(), digest)
        cfg = config()
        self.assertEqual((cfg.batch_size, cfg.accumulate_grad_batches, cfg.num_epochs), (16, 1, 20))
        self.assertEqual(cfg.encode_chunk_size, 256)
        self.assertEqual(cfg.ema["decay"], 0.99)
        self.assertFalse(cfg.ema["switch_ema"])
        self.assertFalse(cfg.ema["exclude_buffers"])
        self.assertEqual(cfg.num_classes, 12)
        self.assertEqual(cfg.medial_route_labels, ["MCL", "Medial Meniscus", "Medial OA"])
        self.assertEqual(cfg.lateral_route_labels, ["Lateral Meniscus", "Lateral OA"])
        self.assertEqual(cfg.physical_fov_mm, 130)
        self.assertEqual(cfg.compartment_image_width, 160)
        self.assertNotIn("/mnt/", json.dumps(cfg.__dict__))
        self.assertNotIn("/home/ian", (ROOT / "maxvit/recipe.json").read_text())

    def test_soft_labels_and_gold_override(self):
        labels = pd.DataFrame({"ACL": [.3, .7], "MCL": [.9, .2]}, index=["a", "b"])
        truth = pd.DataFrame({"ACL": [1., np.nan], "MCL": [0., np.nan]}, index=["a", "b"])
        targets, gold = effective_targets(labels, truth, ["ACL", "MCL"])
        np.testing.assert_array_equal(targets.loc["a"], [1, 0])
        np.testing.assert_array_equal(targets.loc["b"], [.7, .2])
        self.assertEqual(gold.tolist(), [True, False])
        truth.loc["b", "ACL"] = 0
        with self.assertRaisesRegex(ValueError, "Partially"):
            effective_targets(labels, truth, ["ACL", "MCL"])

    def test_fold_groups_and_disjoint_studies(self):
        from skp.datasets.rsna_knee_png import select_studies_for_mode
        frame = pd.read_csv(ROOT / "assets/data/labels_1_85.csv")
        frame["cv_fold"] = frame.fold
        frame["always_validation"] = 0
        self.assertTrue((frame.groupby("fold_group").fold.nunique() == 1).all())
        gold_seen = 0
        for fold in range(5):
            train = select_studies_for_mode(frame, mode="train", fold=fold)
            val = select_studies_for_mode(frame, mode="val", fold=fold)
            self.assertFalse(set(train.StudyInstanceUID) & set(val.StudyInstanceUID))
            self.assertEqual(len(train) + len(val), 4407)
            gold_seen += int(val.is_gold.sum())
        self.assertEqual(gold_seen, 58)

    def test_schedule_counts_optimizer_updates(self):
        from skp.optim import get_scheduler
        cfg = config()
        cfg.world_size, cfg.args = 1, {"strategy": "ddp"}
        # Stub only the loader count; use the actual shipped scheduler implementation.
        for accumulation, total in [(1, 4400), (2, 2200)]:
            cfg.accumulate_grad_batches = accumulation
            with patch("skp.optim.training_batches_per_epoch", return_value=220, create=True):
                with patch("skp.tasks.utils.training_batches_per_epoch", return_value=220):
                    opt = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=2e-4)
                    scheduler = get_scheduler(cfg, opt, train_dataset=object())
            self.assertEqual(scheduler.total_steps, total)
            self.assertEqual(scheduler.warmup_steps, total // 10)
            scheduler.last_epoch = scheduler.warmup_steps
            self.assertAlmostEqual(scheduler.get_lr()[0], 2e-4)
            scheduler.last_epoch = scheduler.total_steps
            self.assertAlmostEqual(scheduler.get_lr()[0], 1e-6)

    def test_ema_warmup_and_buffers(self):
        from skp.callbacks.ema import ModelEmaV3
        model = torch.nn.BatchNorm1d(2)
        ema = ModelEmaV3(model, decay=.99, use_warmup=True)
        self.assertEqual(ema.get_decay(1), 0)
        self.assertAlmostEqual(ema.get_decay(4400), .99)
        model.running_mean.fill_(3)
        model.num_batches_tracked.fill_(5)
        ema.update(model, step=1)
        torch.testing.assert_close(ema.module.running_mean, model.running_mean)
        self.assertEqual(ema.module.num_batches_tracked.item(), 5)

    def test_compartment_geometry(self):
        from skp.models.classification.rsna_knee_maxvit_rect_router import Net
        model = Net.__new__(Net)
        torch.nn.Module.__init__(model)
        model.compartment_fraction, model.square_size, model.rect_width = .65, 256, 160
        volume = torch.arange(256).float().view(1, 1, 1, 1, 256).expand(1, 32, 3, 256, 256)
        medial = model._compartment_view(volume, axis=1, lateral=False)
        lateral = model._compartment_view(volume, axis=1, lateral=True)
        self.assertEqual(tuple(medial.shape), (1, 32, 3, 256, 160))
        self.assertLess(medial.mean(), lateral.mean())
        volume = torch.arange(32).float().view(1, 32, 1, 1, 1).expand(1, 32, 3, 256, 256)
        medial = model._compartment_view(volume, axis=0, lateral=False)
        lateral = model._compartment_view(volume, axis=0, lateral=True)
        self.assertEqual(tuple(lateral.shape), (1, 32, 3, 256, 160))
        self.assertAlmostEqual(medial[0, -1].mean().item(), 20)
        self.assertAlmostEqual(lateral[0, 0].mean().item(), 11)

    def test_runtime_imports(self):
        import skp.train
        from skp.tasks.rsna_knee import Task
        import train_two_view_ema
        self.assertTrue(Path(skp.train.__file__).is_relative_to(ROOT / "maxvit"))
        self.assertTrue(callable(Task))
        self.assertTrue(callable(train_two_view_ema.KneeEMA))

    def test_one_series_per_study_and_actual_schedule(self):
        from skp.tasks.samplers import StudySeriesSampler
        from skp.tasks.utils import training_batches_per_epoch
        cfg = config()
        cfg.world_size, cfg.args = 1, {"strategy": "ddp"}
        dataset = SimpleNamespace(items=[{"study_uid": str(i)} for i in range(3525) for _ in range(3)])
        sampler = StudySeriesSampler(dataset, cfg)
        indices = list(sampler)
        self.assertEqual(len(indices), 3525)
        self.assertEqual(len({dataset.items[i]["study_uid"] for i in indices}), 3525)
        self.assertEqual(training_batches_per_epoch(cfg, dataset), 220)

    def test_study_aggregation_and_gold_threshold_policy(self):
        from skp.metrics.rsna_knee import ValidationMetrics
        cfg = config()
        metric = ValidationMetrics(cfg)
        target = torch.tensor([0, 1, 0, 1, .19, .2, .5, .8])[:, None].expand(8, 12)
        p = torch.linspace(.1, .9, 8)[:, None].expand(8, 12)
        logits = torch.logit(p)
        batch = {"y": target, "y_mask": torch.ones_like(target, dtype=torch.bool),
                 "uncertain_mask": torch.zeros_like(target, dtype=torch.bool),
                 "is_expert": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]),
                 "study_index": torch.arange(8), "fluid_sensitive": torch.ones(8, dtype=torch.bool)}
        metric.update({"logits": logits}, batch)
        metric.update({"logits": logits + 2}, batch)
        averaged_probability = (logits.sigmoid() + (logits + 2).sigmoid()) / 2
        result = metric.compute()
        self.assertEqual(int(result["combined/sample_count"]), 8)
        self.assertEqual(int(result["expert/sample_count"]), 4)
        self.assertAlmostEqual(float(result["combined/loss"]),
            torch.nn.functional.binary_cross_entropy(averaged_probability, target).item(), places=6)
        from sklearn.metrics import roc_auc_score
        self.assertAlmostEqual(float(result["expert/competition_macro_auc"]), roc_auc_score(target[:4, 0], p[:4, 0]))
        for threshold in [.2, .5, .8]:
            truth = target[:, 0].clone()
            truth[4:] = (truth[4:] >= threshold).float()
            self.assertAlmostEqual(float(result[f"combined/competition_macro_auc_threshold_{threshold}"]),
                                   roc_auc_score(truth, p[:, 0]), places=6)

    def test_ema_checkpoint_keeps_raw_and_ema_separate(self):
        from train_two_view_ema import KneeEMA
        from skp.callbacks.ema import ModelEmaV3
        model = torch.nn.Linear(1, 1)
        callback = KneeEMA(decay=.99)
        callback.ema = ModelEmaV3(model, decay=.99)
        with torch.no_grad():
            callback.ema.module.weight.fill_(2)
            model.weight.fill_(5)
        checkpoint = {}
        callback.on_save_checkpoint(None, model, checkpoint)
        self.assertEqual(checkpoint["prediction_weights"], "ema")
        self.assertEqual(checkpoint["state_dict"]["weight"].item(), 2)
        self.assertEqual(checkpoint["raw_training_state_dict"]["weight"].item(), 5)
        callback.on_load_checkpoint(None, model, checkpoint)
        self.assertEqual(checkpoint["state_dict"]["weight"].item(), 5)

    def test_pool_requires_final_heldout_ema_and_valid_hashes(self):
        from maxvit.cli import pool, sha
        from skp.metrics.rsna_knee import _metric_name
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = config()
            cfg.folds_csv = str(root / "folds.csv")
            cfg.dataset_manifest = str(root / "dataset.json")
            Path(cfg.dataset_manifest).write_text('{"components": []}')
            pd.DataFrame({"StudyInstanceUID": [f"s{f}" for f in range(5)], "cv_fold": range(5)}).to_csv(cfg.folds_csv, index=False)
            directories = []
            for fold in range(5):
                directory = root / str(fold)
                directory.mkdir()
                path = directory / "studies_all_series.csv"
                pd.DataFrame({"StudyInstanceUID": [f"s{fold}"], **{
                    "pred_" + _metric_name(name): [.5] for name in cfg.class_names}}).to_csv(path, index=False)
                (directory / "manifest.json").write_text(json.dumps({"fold": fold, "epoch": 19,
                    "prediction_weights": "ema", "global_step": 4400, "class_names": cfg.class_names,
                    "dataset_manifest_sha256": sha(cfg.dataset_manifest), "artifacts_sha256": {path.name: sha(path)}}))
                directories.append(directory)
            with patch("maxvit.cli.config", return_value=cfg):
                pool(SimpleNamespace(oof_dirs=directories, output=root / "pooled.csv"))
                self.assertEqual(len(pd.read_csv(root / "pooled.csv")), 5)
                with self.assertRaisesRegex(ValueError, "Wrong fold"):
                    pool(SimpleNamespace(oof_dirs=[directories[0]] * 5, output=root / "bad.csv"))
                (directories[0] / "studies_all_series.csv").write_text("tampered\n")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    pool(SimpleNamespace(oof_dirs=directories, output=root / "bad.csv"))


if __name__ == "__main__":
    unittest.main()
