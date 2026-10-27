import argparse
from copy import copy
import importlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .artifacts import resolve_artifacts
from .config import setup_sys_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--moaks-index', action='store_true')
    args = parser.parse_args()
    setup_sys_path()
    from ds_knee import SLOTS, build_image_cache, build_slot_table
    from ds_knee_extra import build_extra_cache
    from .prepare_knee_triplet import build_cache
    cfg = copy(importlib.import_module(args.config).cfg)
    cfg.cache_workers = args.workers
    cfg.train_df = str(Path(__file__).resolve().parents[2] / 'assets/data/train_1_85.csv')
    resolve_artifacts(cfg)
    cv2.setNumThreads(0)
    data = Path(__file__).resolve().parents[2] / 'assets/data'
    base = pd.read_csv(data / 'labels_1_85.csv')
    primary = build_slot_table(cfg)
    primary['slot_id'] = primary.slot.map({slot: index for index, slot in enumerate(SLOTS)})
    all_series = build_slot_table(cfg, keep_duplicates=True)
    extra = all_series.loc[~all_series.SeriesInstanceUID.isin(primary.SeriesInstanceUID)].copy()
    extra = extra.sort_values(['StudyInstanceUID', 'output_n_slices', 'SeriesInstanceUID'],
                              ascending=[True, False, True])
    extra['extra_rank'] = extra.groupby('StudyInstanceUID').cumcount()
    extra = extra.loc[extra.extra_rank < 4].copy()
    extra['cache_row'] = np.arange(len(extra))
    extra.to_csv(getattr(cfg, 'extra_slots_csv', data / 'selected_extra_slots_8_6.csv'), index=False)
    native_extra = extra.copy()
    native_extra['slot_id'] = native_extra.extra_rank + len(SLOTS)
    selected = pd.concat([primary, native_extra], ignore_index=True)
    selected.to_csv(getattr(cfg, 'selected_slots_csv', data / 'selected_slots_8_13.csv'), index=False)
    if args.moaks_index:
        rows = {uid: row for row, uid in enumerate(base.StudyInstanceUID)}
        slots = dict(zip(primary.SeriesInstanceUID, primary.slot_id))
        all_series['cache_hit'] = all_series.SeriesInstanceUID.isin(slots)
        all_series['cache_row'] = all_series.StudyInstanceUID.map(rows)
        all_series['cache_slot'] = all_series.SeriesInstanceUID.map(slots).fillna(-1).astype(int)
        from .artifacts import study_lateralities
        sides = pd.Series(study_lateralities(cfg), dtype=object)
        all_series['laterality'] = all_series.StudyInstanceUID.map(sides)
        all_series.to_csv(data / 'competition_series_moaks.csv', index=False)
        return
    resolve_artifacts(cfg)
    if cfg.dataset == 'ds_knee_triplet':
        print(build_cache(cfg, args.workers))
    else:
        build_image_cache(cfg, base.StudyInstanceUID.tolist(), primary)
        if cfg.dataset in ('ds_knee_extra', 'ds_knee_cached_triplet'):
            build_extra_cache(cfg)


if __name__ == '__main__':
    main()
