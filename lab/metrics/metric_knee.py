from pathlib import Path
import os
import tempfile
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

def macro_auc(truth, probabilities):
    per = {}
    for index, target in enumerate(TARGETS):
        col = truth[:, index]
        ok = ~np.isnan(col)
        per[target] = float(roc_auc_score(col[ok], probabilities[ok, index])) if len(np.unique(col[ok])) >= 2 else float('nan')
    values = [value for value in per.values() if np.isfinite(value)]
    return (float(np.mean(values)) if values else float('nan'), per)

def bce(truth, probabilities):
    probabilities = np.clip(probabilities, 1e-06, 1 - 1e-06)
    return -(truth * np.log(probabilities) + (1 - truth) * np.log(1 - probabilities))

def calc_metric(cfg, pp_out, val_df, pre='val'):
    probabilities = pp_out['predictions'].detach().cpu().numpy()
    auxiliary = pp_out['aux_predictions'].detach().cpu().numpy()
    truth = val_df[[f'bin_{t}' for t in TARGETS]].to_numpy(np.float32)
    soft = val_df[[f'soft_{t}' for t in TARGETS]].to_numpy(np.float32)
    conf = val_df[[f'conf_{t}' for t in TARGETS]].to_numpy(np.float32)
    if not cfg.use_conf_weight:
        conf = np.ones_like(soft)
    gold = val_df.is_gold.to_numpy().astype(bool)
    overall, per = macro_auc(truth, probabilities)
    threshold = getattr(cfg, 'pseudo_auc_threshold', None)
    pseudo_truth = truth if threshold is None else (soft >= threshold).astype(np.float32)
    pseudo, _ = macro_auc(pseudo_truth[~gold], probabilities[~gold])
    gold_auc, _ = macro_auc(truth[gold], probabilities[gold])
    pseudo_loss = (conf[~gold] * bce(soft[~gold], probabilities[~gold])).mean()
    pseudo_loss += cfg.aux_weight * (conf[~gold] * bce(soft[~gold], auxiliary[~gold])).mean()
    return {'auc_overall': overall, 'auc_pseudo': pseudo, 'auc_gold': gold_auc, 'loss_pseudo': float(pseudo_loss), 'loss_gold': float(bce(truth[gold], probabilities[gold]).mean()), **{'auc_' + target.replace("'", ''): value for target, value in per.items()}}

def resume_history(cfg, completed_epochs):
    path = Path(cfg.output_dir) / f'fold{cfg.fold}' / f'history_seed{cfg.seed}.csv'
    if path.exists():
        history = pd.read_csv(path)
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.csv', delete=False) as file:
            history.loc[history.epoch <= completed_epochs].to_csv(file, index=False)
            temporary = file.name
        os.replace(temporary, path)

def summarize_epoch(cfg, raw, ema, model, epoch):
    use_ema = ema is not None and cfg.report_ema
    selected = ema if use_ema else raw
    variant = 'ema' if use_ema else 'model'
    row = {'fold': cfg.fold, 'epoch': epoch + 1, 'train_loss': cfg.last_train_losses['loss'], 'variant': variant, 'epoch_seconds': time.monotonic() - cfg.epoch_started, **{f'val_{key}': value for key, value in selected.items()}, **{f'raw_{key}': value for key, value in raw.items()}, **{f'ema_{key}': value for key, value in (ema or {}).items()}}
    net = model.module if hasattr(model, 'module') else model
    gates = {}
    if cfg.slice_attn and net.slice_gate is not None:
        gates['slice_gate_abs'] = float(net.slice_gate.detach().abs().mean())
    if cfg.spatial_attn and net.spatial_gate is not None:
        gates['spatial_gate_abs'] = float(net.spatial_gate.detach().abs().mean())
    row.update(gates)
    path = Path(cfg.output_dir) / f'fold{cfg.fold}' / f'history_seed{cfg.seed}.csv'
    pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
    gate_text = ''
    if 'slice_gate_abs' in gates:
        gate_text += f'|gate|={row['slice_gate_abs']:.4f} '
    if 'spatial_gate_abs' in gates:
        gate_text += f'|spatial_gate|={row['spatial_gate_abs']:.4f} '
    print(f'fold {cfg.fold} ep {epoch + 1:02d}/{cfg.epochs} loss={row['train_loss']:.4f}/{selected['loss_pseudo']:.4f} pseudo={selected['auc_pseudo']:.5f} gold={selected['auc_gold']:.5f} overall={selected['auc_overall']:.5f} {gate_text}-> {variant} ({row['epoch_seconds']:.1f}s)', flush=True)
    return selected
