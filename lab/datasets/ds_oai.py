import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ds_knee import augment_study, batch_to_device
from utils.prepare_oai import sha256


def prepare_data(cfg):
    for path, expected in [(cfg.train_df, cfg.train_df_sha256),
                           (cfg.oai_labels, cfg.oai_labels_sha256),
                           (cfg.oai_schema, cfg.oai_schema_sha256)]:
        if sha256(path) != expected:
            raise ValueError(f"Pinned OAI artifact differs: {path}")
    frame = pd.read_csv(cfg.train_df, dtype={"patient_id": str})
    series = pd.read_csv(cfg.oai_series_csv, dtype={"patient_id": str})
    available = set(series.case_id)
    if cfg.oai_smoke_cases:
        if cfg.save_checkpoint:
            raise ValueError("Partial OAI smoke runs require -D")
        frame = frame.loc[frame.case_id.isin(available)].groupby("fold", group_keys=False).head(cfg.oai_smoke_cases)
    elif not set(frame.case_id).issubset(available):
        raise ValueError("The frozen OAI cohort is not fully cached")
    training = frame.loc[frame.fold.ne(cfg.fold)].copy()
    validation = frame.loc[frame.fold.eq(cfg.fold)].merge(series[["case_id", "cache_path"]], on="case_id")
    return training, validation, None


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = mode == "train"
        series = pd.read_csv(cfg.oai_series_csv)
        self.paths = series.groupby("case_id").cache_path.apply(list).to_dict()
        with np.load(cfg.oai_labels) as labels:
            indices = self.df.target_row.to_numpy(int)
            self.targets = labels["targets"][indices]
            self.target_masks = labels["mask"][indices]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.train:
            paths = self.paths[row.case_id]
            path = paths[np.random.randint(len(paths))]
        else:
            path = row.cache_path
        with np.load(path) as cache:
            images = cache["images"][None, :, None]
            smask = cache["smask"][None]
        mask = np.ones(1, np.float32)
        if self.train:
            images, mask = augment_study(self.cfg, images, mask)
        return {"images": torch.from_numpy(images), "mask": torch.from_numpy(mask),
                "smask": torch.from_numpy(smask), "target": torch.from_numpy(self.targets[index]),
                "target_mask": torch.from_numpy(self.target_masks[index])}


tr_collate_fn = None
val_collate_fn = None
