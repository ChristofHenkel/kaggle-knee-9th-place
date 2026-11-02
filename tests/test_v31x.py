import argparse
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


class V31xTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lab.utils.config import setup_sys_path
        setup_sys_path()

    def test_shared_labels_and_configs(self):
        labels_path = ROOT / "assets/data/labels_1_85.csv"
        self.assertEqual(
            hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "e9018032ee2eab75df52ff3a69dbf413556b3c51094146542deb1905597af692")
        labels = pd.read_csv(
            labels_path, dtype={"StudyInstanceUID": str})
        self.assertEqual(len(labels), 4407)
        self.assertEqual(labels.StudyInstanceUID.nunique(), 4407)
        self.assertEqual(int(labels.is_gold.sum()), 58)
        self.assertEqual(
            labels.fold.value_counts().sort_index().to_dict(),
            {0: 882, 1: 882, 2: 881, 3: 881, 4: 881})

        expected = {
            "v307": ("convnext_2p5d", 14, 2026, 0, True),
            "v310": ("efficientnet_2p5d", 12, 2026, 0, True),
            "v311": ("efficientnet_2p5d", 19, 3026, -1, False),
            "v313": ("convnext_2p5d", 21, 4026, -1, False),
        }
        for experiment, (
                encoder, epochs, seed, fold, validation) in expected.items():
            cfg = importlib.import_module(f"cfg_{experiment}").cfg
            self.assertEqual(cfg.encoder_3d, encoder)
            self.assertEqual(cfg.epochs, epochs)
            self.assertEqual(cfg.schedule_epochs, 25)
            self.assertEqual(cfg.seed, seed)
            self.assertEqual(cfg.fold, fold)
            self.assertEqual(cfg.val, validation)
            self.assertEqual(cfg.checkpoint_keep_every, 0)
            self.assertFalse(cfg.channels_last)
            self.assertEqual(cfg.batch_size, 2)
            self.assertEqual(cfg.grad_accumulation, 2)
            self.assertEqual(cfg.sampler_batch_size, 4)
            if validation:
                self.assertEqual(cfg.eval_epochs, epochs)
                self.assertTrue(cfg.save_val_data)
                self.assertTrue(cfg.seed_fold_offset)
            else:
                self.assertFalse(cfg.save_val_data)
                self.assertFalse(cfg.seed_fold_offset)

    def test_depth_mixer_starts_as_live_noop(self):
        from mdl_knee_2p5d import DepthMixer

        cfg = SimpleNamespace(
            hybrid_bottleneck=8, hybrid_depth_kernel=3,
            hybrid_gate_init=1.0, type_embedding=True,
            slot_names=("A", "B"))
        mixer = DepthMixer(cfg, channels=4)
        feature = torch.randn(2, 4, 3, 2, 2, requires_grad=True)
        output = mixer(feature, torch.tensor([0, 1]))
        torch.testing.assert_close(output, feature)
        output.sum().backward()
        self.assertGreater(
            float(mixer.depth_out.weight.grad.abs().sum()), 0.0)

    def test_scheduler_uses_25_epoch_horizon(self):
        from lab.utils.scheduler import get_scheduler

        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        cfg = SimpleNamespace(
            schedule="parag_cosine", optimizer_steps_per_epoch=10,
            epochs=19, schedule_epochs=25, batch_size=4,
            grad_accumulation=1, warmup_frac=0.15, min_lr_frac=0.02)
        scheduler = get_scheduler(cfg, optimizer, total_steps=4407)
        scale = scheduler.lr_lambdas[0]
        self.assertGreater(scale(190), cfg.min_lr_frac)
        self.assertAlmostEqual(scale(250), cfg.min_lr_frac)

    def test_accumulation_preserves_effective_batch_steps(self):
        from lab.utils.data import DistributedSizeBalancedSampler

        sampler = DistributedSizeBalancedSampler(
            sizes=np.arange(4407),
            weights=np.ones(4407),
            batch_size=4,
            num_replicas=1,
            rank=0,
            seed=2026,
        )
        self.assertEqual(len(sampler), 4404)
        physical_batches = len(sampler) // 2
        self.assertEqual(physical_batches // 2, 1101)
        self.assertEqual(len(list(iter(sampler))), 4404)

    def test_repeat_maps_to_configured_seed(self):
        from lab.utils.prepare_v31x import command_for_train

        args = argparse.Namespace(
            experiment="v313", repeat=5, fold=None,
            nproc=1, master_port=None,
            output_dir=None, checkpoint=None)
        command = command_for_train(args)
        self.assertEqual(command[command.index("--seed") + 1], "4030")
        self.assertEqual(command[command.index("--fold") + 1], "-1")

    def test_cv_command_uses_fold_seed_offset(self):
        from lab.utils.prepare_v31x import command_for_train

        args = argparse.Namespace(
            experiment="v310", repeat=None, fold=4,
            nproc=1, master_port=None,
            output_dir=None, checkpoint=None)
        command = command_for_train(args)
        self.assertEqual(command[command.index("--fold") + 1], "4")
        self.assertNotIn("--seed", command)

    def test_cache_command_uses_converter(self):
        from lab.utils.prepare_v31x import command_for_cache

        args = argparse.Namespace(
            experiment="v307", workers=16, laterality_workers=8,
            input_root="/competition", study_limit=2)
        command = command_for_cache(args)
        self.assertIn(
            "scripts/dicom_series_to_processed_npz.py", command)
        self.assertEqual(
            command[command.index("--input-root") + 1], "/competition")
        self.assertEqual(
            command[command.index("--study-limit") + 1], "2")

    def test_cached_shards_loader_densifies_slots(self):
        from ds_knee import TARGETS
        from ds_knee_cached_shards import CustomDataset

        uid = "study"
        slots = ("AXI_FS", "COR_FS", "SAG_FS")
        frame = pd.DataFrame([{
            "StudyInstanceUID": uid,
            "fold_group": "group",
            **{f"soft_{target}": 0.5 for target in TARGETS},
            **{f"conf_{target}": 1.0 for target in TARGETS},
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = np.arange(
                2 * 2 * 3 * 4 * 4, dtype=np.uint8
            ).reshape(2, 2, 3, 4, 4)
            np.savez(
                root / f"{uid}.npz",
                imgs=images,
                smask=np.ones((2, 2), np.float32),
                slot_names=np.asarray(("AXI_FS", "SAG_FS")),
                slot_ids=np.asarray((0, 2), np.int64),
                series_uid=np.asarray(("series0", "series2")),
                n_errors=np.int64(0),
            )
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "image_shape": [2, 3, 4, 4],
                "slots": list(slots),
                "studies": [{
                    "StudyInstanceUID": uid, "status": "ok",
                    "n_slots": 2, "n_errors": 0,
                }],
            }))
            cfg = SimpleNamespace(
                cached_shards_dir=str(root), name="cfg_v307",
                slot_names=slots, slices_per_slot=2,
                input_channels=3, img_px=4)
            dataset = CustomDataset(frame, cfg, mode="val")
            item = dataset[0]
            self.assertEqual(tuple(item["images"].shape), (3, 2, 3, 4, 4))
            self.assertEqual(item["mask"].tolist(), [1.0, 0.0, 1.0])
            self.assertTrue(torch.equal(
                item["images"][0], torch.from_numpy(images[0])))
            self.assertTrue(torch.equal(
                item["images"][2], torch.from_numpy(images[1])))
            self.assertEqual(dataset.sample_sizes.tolist(), [2])

    def test_collects_only_final_ema_oofs(self):
        from lab.utils.prepare_v31x import collect_oof, write_fusion

        cfg = importlib.import_module("cfg_v307").cfg
        saved_train_df = cfg.train_df
        labels = pd.read_csv(
            ROOT / "assets/data/labels_1_85.csv",
            dtype={"StudyInstanceUID": str})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            labels.to_csv(train_path, index=False)
            cfg.train_df = str(train_path)
            try:
                for fold, rows in {
                        0: 882, 1: 882, 2: 881, 3: 881, 4: 881}.items():
                    seed = 2026 + fold
                    folder = root / f"fold{fold}"
                    folder.mkdir()
                    probability = (fold + 1) / 10
                    torch.save(
                        {"predictions": torch.full(
                            (rows, 12), probability)},
                        folder / f"val_ema_data_seed{seed}.pth")
                    torch.save(
                        {"epoch": 13, "ema": {}},
                        folder / f"checkpoint_last_seed{seed}.pth")
                    pd.DataFrame(
                        [{"epoch": 14, "variant": "ema"}]
                    ).to_csv(
                        folder / f"history_seed{seed}.csv", index=False)
                output = root / "final_oof.csv"
                collect_oof(argparse.Namespace(
                    experiment="v307", run_root=str(root),
                    output=str(output), force=False))
            finally:
                cfg.train_df = saved_train_df

            pooled = pd.read_csv(output)
            self.assertEqual(len(pooled), 4407)
            for fold in range(5):
                values = pooled.loc[
                    pooled.fold.eq(fold), "ACL"].to_numpy()
                np.testing.assert_allclose(values, (fold + 1) / 10)
            manifest = json.loads(
                output.with_suffix(".provenance.json").read_text())
            self.assertEqual(manifest["epoch"], 14)
            self.assertEqual(manifest["variant"], "ema")
            fused_path = root / "labels_v313.csv"
            fused = write_fusion(
                "v313", ROOT / "assets/data/labels_1_85.csv",
                output, fused_path, force=False)
            source = pd.read_csv(
                ROOT / "assets/data/labels_1_85.csv",
                dtype={"StudyInstanceUID": str}).set_index(
                    "StudyInstanceUID")
            non_gold_uid = fused.loc[fused.is_gold.eq(0)].index[0]
            fold = int(fused.loc[non_gold_uid, "fold"])
            self.assertAlmostEqual(
                fused.loc[non_gold_uid, "ACL"],
                0.5 * source.loc[non_gold_uid, "ACL"]
                + 0.5 * (fold + 1) / 10)
            gold_uid = fused.loc[fused.is_gold.eq(1)].index[0]
            self.assertEqual(
                fused.loc[gold_uid, "ACL"],
                source.loc[gold_uid, "ACL"])


if __name__ == "__main__":
    unittest.main()
