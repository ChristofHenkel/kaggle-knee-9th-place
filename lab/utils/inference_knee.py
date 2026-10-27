from copy import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .artifacts import sha256
from .config import setup_sys_path


@torch.inference_mode()
def predict(config, weights, competition, output, gpu=0, workers=3):
    setup_sys_path()
    from ds_knee_inference import DicomDataset, TARGETS

    cfg = copy(importlib.import_module(config).cfg)
    cfg.encoder_3d_pretrained = False
    weights, competition, output = Path(weights), Path(competition), Path(output)
    manifest = json.loads((weights / 'manifest.json').read_text())
    if manifest['config'] != config:
        raise ValueError('Checkpoint bundle and requested config differ')
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    sample = pd.read_csv(competition / 'sample_submission.csv')
    dataset = DicomDataset(sample.StudyInstanceUID.tolist(), cfg, competition, 'test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    scores = np.zeros((len(sample), len(TARGETS)), dtype=np.float64)
    Net = importlib.import_module(cfg.model).Net
    for member in manifest['members']:
        path = weights / member['file']
        if sha256(path) != member['sha256']:
            raise ValueError(f'Checkpoint checksum differs: {path}')
        model = Net(cfg)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'], strict=True)
        model = model.eval().to(device)
        predictions = []
        for batch in loader:
            images = {key: batch[key].to(device) for key in ('images', 'mask', 'smask')}
            with torch.autocast('cuda', dtype=torch.float16):
                values = model(images)['predictions'].float().cpu().numpy()
            values[~batch['mask'].any(dim=1).numpy()] = .5
            predictions.append(values)
        scores += np.concatenate(predictions) / len(manifest['members'])
        del model
    result = pd.DataFrame(scores, columns=TARGETS)
    result.insert(0, 'StudyInstanceUID', sample.StudyInstanceUID.to_numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
