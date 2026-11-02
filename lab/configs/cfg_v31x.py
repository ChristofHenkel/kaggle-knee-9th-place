from copy import copy
from pathlib import Path

from cfg_8_29 import cfg as parent_cfg


SLOTS = (
    "AXI_FS", "AXI_NFS",
    "COR_FS", "COR_NFS_FL", "COR_NFS_ST",
    "SAG_FS", "SAG_NFS_FL", "SAG_NFS_ST",
    "EXTRA_00", "EXTRA_01", "EXTRA_02", "EXTRA_03",
)

EXPERIMENTS = {
    "v307": {
        "kind": "cv",
        "epochs": 14,
        "seed": 2026,
        "encoder_3d": "convnext_2p5d",
        "encoder_2d": "convnext_tiny.fb_in22k_ft_in1k",
        "encoder_repo_id": "timm/convnext_tiny.fb_in22k_ft_in1k",
        "encoder_revision": "bc48a87f119bae1b06418f5340bfa97ef75d4609",
        "encoder_weights_name": "convnext_tiny.fb_in22k_ft_in1k",
        "encoder_weights_sha256":
            "77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a",
        "label_name": "1_85",
        "label_sha256":
            "e9018032ee2eab75df52ff3a69dbf413556b3c51094146542deb1905597af692",
    },
    "v310": {
        "kind": "cv",
        "epochs": 12,
        "seed": 2026,
        "encoder_3d": "efficientnet_2p5d",
        "encoder_2d": "tf_efficientnetv2_s",
        "encoder_repo_id": "timm/tf_efficientnetv2_s.in21k_ft_in1k",
        "encoder_revision": "ea9abc143ea2b9d8e1ec1de277bce02149b9cf0e",
        "encoder_weights_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "encoder_weights_sha256":
            "6f1933fb6c0d760eae03863ad0110570393e16c0610f8fe94dc9f978e52ec59c",
        "label_name": "1_85",
        "label_sha256":
            "e9018032ee2eab75df52ff3a69dbf413556b3c51094146542deb1905597af692",
    },
    "v311": {
        "kind": "fullfit",
        "epochs": 19,
        "seed": 3026,
        "encoder_3d": "efficientnet_2p5d",
        "encoder_2d": "tf_efficientnetv2_s",
        "encoder_repo_id": "timm/tf_efficientnetv2_s.in21k_ft_in1k",
        "encoder_revision": "ea9abc143ea2b9d8e1ec1de277bce02149b9cf0e",
        "encoder_weights_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "encoder_weights_sha256":
            "6f1933fb6c0d760eae03863ad0110570393e16c0610f8fe94dc9f978e52ec59c",
        "label_name": "v311",
        "label_sha256": None,
    },
    "v313": {
        "kind": "fullfit",
        "epochs": 21,
        "seed": 4026,
        "encoder_3d": "convnext_2p5d",
        "encoder_2d": "convnext_tiny.fb_in22k_ft_in1k",
        "encoder_repo_id": "timm/convnext_tiny.fb_in22k_ft_in1k",
        "encoder_revision": "bc48a87f119bae1b06418f5340bfa97ef75d4609",
        "encoder_weights_name": "convnext_tiny.fb_in22k_ft_in1k",
        "encoder_weights_sha256":
            "77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a",
        "label_name": "v313",
        "label_sha256": None,
    },
}


def build_config(experiment):
    spec = EXPERIMENTS[experiment]
    root = Path(__file__).resolve().parents[2]
    cfg = copy(parent_cfg)
    cfg.name = f"cfg_{experiment}"
    cfg.parent = "cfg_8_29"
    cfg.output_dir = str(root / "outputs" / cfg.name)
    cfg.train_df = str(root / f"assets/data/train_{experiment}.csv")
    cfg.label_csv = str(
        root / f"assets/data/labels_{spec['label_name']}.csv")
    cfg.label_sha256 = spec["label_sha256"]
    cfg.dataset = "ds_knee_cached_shards"
    cfg.cached_shards_dir = str(root / "assets/data/cached_shards")
    cfg.cache_dir = cfg.cached_shards_dir
    cfg.selected_slots_csv = None
    cfg.extra_slots_csv = None
    cfg.image_metadata_csv = None
    cfg.image_metadata_sha256 = None
    cfg.train = True
    cfg.val = spec["kind"] == "cv"
    cfg.test = False
    cfg.train_val = False
    cfg.fold = 0 if spec["kind"] == "cv" else -1
    cfg.seed = spec["seed"]
    cfg.seed_fold_offset = spec["kind"] == "cv"
    cfg.epochs = spec["epochs"]
    cfg.eval_epochs = spec["epochs"]
    cfg.schedule_epochs = 25
    cfg.encoder_3d = spec["encoder_3d"]
    cfg.encoder_2d = spec["encoder_2d"]
    cfg.encoder_3d_pretrained = True
    cfg.encoder_repo_id = spec["encoder_repo_id"]
    cfg.encoder_revision = spec["encoder_revision"]
    cfg.encoder_weights = str(
        root / "assets/weights/pretrained/timm"
        / spec["encoder_weights_name"] / "model.safetensors")
    cfg.encoder_weights_sha256 = spec["encoder_weights_sha256"]
    cfg.gradient_checkpointing = True
    cfg.backbone_parameter_prefix = "encoder.backbone."
    cfg.hybrid_bottleneck = 384
    cfg.hybrid_depth_kernel = 3
    cfg.hybrid_gate_init = 1.0
    cfg.type_embedding = True

    cfg.slot_names = SLOTS
    cfg.slices_per_slot = 32
    cfg.cache_slices = 32
    cfg.img_px = 320
    cfg.crop_mm = 110.0
    cfg.use_acquisition_metadata = True
    cfg.normalize_laterality = True
    cfg.laterality_workers = 1
    cfg.input_channels = 3
    cfg.native_triplets = True
    cfg.triplet_delta_mm = 3.0
    cfg.input_normalization = "unit"
    cfg.channels_last = False

    cfg.batch_size = 2
    cfg.batch_size_val = 2
    cfg.grad_accumulation = 2
    cfg.sampler_batch_size = 4
    cfg.optimizer = "AdamW"
    cfg.lr = 1.0e-3
    cfg.lr_backbone = 1.5e-4
    cfg.lr_head = 1.0e-3
    cfg.weight_decay = 1.0e-4
    cfg.schedule = "parag_cosine"
    cfg.schedule_on_optimizer_step = True
    cfg.warmup_frac = 0.15
    cfg.min_lr_frac = 0.02
    cfg.clip_grad = 1.0
    cfg.ema_decay = 0.999
    cfg.ema_warmup = False

    cfg.loss_weight_csv = None
    cfg.use_conf_weight = False
    cfg.gold_weight = 1.0
    cfg.coarse_dropout_prob = 0.0
    cfg.balanced_batches = True
    cfg.expected_world_size = 1
    cfg.save_val_data = spec["kind"] == "cv"
    cfg.save_val_snapshots = False
    cfg.save_optimizer_state = True
    cfg.resumable = True
    cfg.save_only_last_ckpt = False
    cfg.checkpoint_keep_every = 0
    cfg.frozen_training_table = True
    cfg.train_df_sha256 = None
    cfg.selected_slots_sha256 = None
    cfg.extra_slots_sha256 = None
    return cfg
