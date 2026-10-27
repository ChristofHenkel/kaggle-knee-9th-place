import hashlib
from pathlib import Path

import pandas as pd


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_pinned_csv(path, expected=None):
    if expected is not None and sha256(path) != expected:
        raise ValueError(f'Artifact checksum mismatch: {path}')
    return pd.read_csv(path)


def resolve_artifacts(cfg):
    path = getattr(cfg, 'image_metadata_csv', None)
    if path is not None and Path(path).is_file():
        read_pinned_csv(path, cfg.image_metadata_sha256)
    for field in ('train_df', 'selected_slots_csv', 'extra_slots_csv'):
        path = getattr(cfg, field, None)
        if path is not None and Path(path).is_file():
            setattr(cfg, field.removesuffix('_csv') + '_sha256', sha256(path))
    for field in ('oai_labels', 'oai_schema', 'oai_selected_archives'):
        path = getattr(cfg, field, None)
        if path is not None and Path(path).is_file():
            setattr(cfg, field + '_sha256', sha256(path))


def read_image_metadata(cfg):
    return read_pinned_csv(cfg.image_metadata_csv, cfg.image_metadata_sha256)


def study_lateralities(cfg):
    if not getattr(cfg, 'normalize_laterality', True):
        return {}
    metadata = read_image_metadata(cfg)
    return metadata.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').training_laterality.to_dict()
