import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from metric_knee import resume_history

def calc_metric(cfg, pp_out, val_df, pre='val'):
    predictions = pp_out['predictions'].cpu().numpy()
    pooled = pd.DataFrame(predictions).groupby(val_df.case_id.to_numpy(), sort=False).mean()
    cases = val_df.drop_duplicates('case_id').set_index('case_id').loc[pooled.index]
    with np.load(cfg.oai_labels) as labels:
        target = labels['targets'][cases.target_row.to_numpy(int)]
        mask = labels['mask'][cases.target_row.to_numpy(int)]
    schema = json.loads(Path(cfg.oai_schema).read_text())
    probabilities = pooled.to_numpy()
    offset = 0
    losses = []
    for field in schema['fields']:
        end = offset + len(field['classes'])
        losses.append(-(target[:, offset:end] * np.log(probabilities[:, offset:end].clip(1e-07, 1))).sum(-1))
        offset = end
    losses = np.stack(losses, axis=1)
    return {'moaks_ce': float(((losses * mask).sum(-1) / mask.sum(-1)).mean())}

def summarize_epoch(cfg, raw, ema, model, epoch):
    row = {'epoch': epoch + 1, 'train_loss': cfg.last_train_losses['loss'], 'val_moaks_ce': raw['moaks_ce'], 'epoch_seconds': time.monotonic() - cfg.epoch_started}
    path = Path(cfg.output_dir) / f'fold{cfg.fold}' / f'history_seed{cfg.seed}.csv'
    pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
    print(f'OAI epoch {epoch + 1}/{cfg.epochs}: train CE={row['train_loss']:.5f}, val CE={row['val_moaks_ce']:.5f}, seconds={row['epoch_seconds']:.1f}', flush=True)
    return raw
