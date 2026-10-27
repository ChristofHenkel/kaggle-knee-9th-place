import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch

from .artifacts import sha256


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'assets/data'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CONF = ['conf_' + target for target in TARGETS]
SPLIT = ['fold', 'fold_group', 'is_gold']
SIX = ['Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion']


def table(path):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    frame.StudyInstanceUID = frame.StudyInstanceUID.str.strip()
    return frame.set_index('StudyInstanceUID', verify_integrity=True)


def canonical():
    return table(DATA / 'labels_1_85.csv')


def probabilities(path, index, targets=TARGETS):
    frame = table(path)
    if set(frame.index) != set(index):
        raise ValueError(f'Study coverage differs: {path}')
    values = frame.loc[index, targets]
    if not np.isfinite(values.to_numpy()).all() or not ((values >= 0) & (values <= 1)).all().all():
        raise ValueError(f'Invalid probabilities: {path}')
    return values


def save(frame, path, sources, recipe):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)
    provenance = {'recipe': recipe, 'sources': {str(p): sha256(p) for p in sources},
                  'output_sha256': sha256(path)}
    path.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(path)


def training_table(labels, competition):
    base = canonical().loc[labels.index]
    truth = table(Path(competition) / 'train.csv').loc[labels.index, TARGETS]
    gold = truth.notna().all(axis=1)
    if not np.array_equal(gold.to_numpy(), base.is_gold.to_numpy(bool)):
        raise ValueError('Official gold membership differs from the frozen split')
    frame = labels[TARGETS].copy()
    frame[CONF] = 1.0
    frame[SPLIT] = base[SPLIT]
    for target in TARGETS:
        values = labels[target].to_numpy(np.float32)
        values[gold] = truth.loc[gold, target].to_numpy(np.float32)
        frame['soft_' + target] = values
        frame['bin_' + target] = (values >= .5).astype(np.float32)
    return frame


def prepare_base(args):
    labels = canonical()
    save(training_table(labels, args.competition), DATA / 'train_1_85.csv',
         [DATA / 'labels_1_85.csv', Path(args.competition) / 'train.csv'], 'L85 with exact gold targets')


def collect_oof(args):
    base = canonical()
    result = pd.DataFrame(index=base.index, columns=TARGETS, dtype=float)
    sources = []
    expected = 25 if args.config == 'cfg_1_85' else 35
    stem = 'val' if args.config == 'cfg_1_85' else 'val_ema'
    root = Path(args.run_root or ROOT / 'outputs' / args.config)
    for fold in range(5):
        folder = root / f'fold{fold}'
        files = list(folder.glob(stem + '_data_seed*.pth'))
        if len(files) != 1:
            raise ValueError(f'Expected one final OOF tensor in {folder}; found {len(files)}')
        path = files[0]
        seed = path.stem.split('seed')[-1]
        history = folder / f'history_seed{seed}.csv'
        if int(pd.read_csv(history).epoch.iloc[-1]) != expected:
            raise ValueError(f'Expected final epoch {expected}: {history}')
        checkpoint = folder / f'checkpoint_last_seed{seed}.pth'
        with torch.serialization.safe_globals([defaultdict, list]):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            if state['epoch'] + 1 != expected:
                raise ValueError(f'Checkpoint has not reached epoch {expected}: {checkpoint}')
            prediction = torch.load(path, map_location='cpu', weights_only=True)['predictions'].numpy()
        held_out = base.fold.eq(fold)
        if prediction.shape != (int(held_out.sum()), len(TARGETS)):
            raise ValueError(f'OOF shape differs from the frozen held-out rows: {path}')
        result.loc[held_out, TARGETS] = prediction
        sources.extend([path, history])
    if not np.isfinite(result.to_numpy()).all() or not ((result >= 0) & (result <= 1)).all().all():
        raise ValueError('Invalid OOF probabilities')
    output = Path(args.output or ROOT / 'outputs' / args.config / 'final_oof.csv')
    save(result, output, sources + [DATA / 'labels_1_85.csv'],
         f'{args.config} final epoch {expected} {"ordinary" if stem == "val" else "EMA"} held-out probabilities')


def fuse(args):
    base = canonical()
    effective = training_table(base, args.competition)
    targets = effective[['soft_' + t for t in TARGETS]].to_numpy()
    oof_path = Path(args.oof or ROOT / 'outputs' / args.source / 'final_oof.csv')
    prediction = probabilities(oof_path, base.index).to_numpy()
    sources = [DATA / 'labels_1_85.csv', Path(args.competition) / 'train.csv', oof_path]
    if args.source == 'cfg_1_85':
        weights = pd.DataFrame(1 - np.abs(targets - prediction), index=base.index, columns=TARGETS)
        save(weights, DATA / 'loss_weights_1_85.csv', sources, '1 - abs(effective_L85 - OOF_1_85)')
        suffix = '1_91'
    else:
        suffix = '8_31'
    labels = base.copy()
    labels[TARGETS] = .5 * targets + .5 * prediction
    gold = base.is_gold.eq(1)
    labels.loc[gold, TARGETS] = targets[gold]
    save(labels, DATA / f'labels_{suffix}.csv', sources,
         f'0.5 * effective_L85 + 0.5 * {args.source} final OOF; exact gold restored')
    save(training_table(labels, args.competition), DATA / f'train_{suffix}.csv', sources,
         f'Effective {suffix} targets; original L85 folds')


def compose(args):
    base = canonical()
    index = base.index
    inputs = [args.opus012, args.opus016, args.sol_review, args.gemini_report,
              args.sol_review_six, args.gemini_images]
    a, b, sol, gemini_report, sol_six, gemini_images = [probabilities(p, index) for p in inputs]
    conf = pd.DataFrame(np.clip(1 - np.abs(a.to_numpy() - b.to_numpy()), .25, 1),
                        index=index, columns=CONF)
    l26 = .5 * a + .5 * b
    matched = pd.DataFrame(index=index, columns=TARGETS, dtype=float)
    for target in TARGETS:
        quantiles = (rankdata(sol[target], method='average') - 1) / (len(index) - 1)
        matched[target] = np.quantile(l26[target], quantiles, method='linear')
    l51 = .5 * l26 + .5 * matched
    l63 = .6 * l51 + .2 * gemini_report + .2 * sol_six
    l77 = l63.copy()
    three = ['Effusion', 'Synovitis', "Baker's"]
    l77[three] = .5 * l63[three] + .5 * gemini_images[three]
    recipes = {'26': l26, '46': sol, '51': l51, '63': l63, '77': l77}
    if args.v700 is not None and args.oa is not None:
        v700, oa = probabilities(args.v700, index, SIX), probabilities(args.oa, index, SIX)
        l85 = l77.copy()
        l85[SIX] = .5 * l77[SIX] + .25 * v700[SIX] + .25 * oa[SIX]
        recipes['85'] = l85
        inputs.extend([args.v700, args.oa])
    for name, values in recipes.items():
        labels = values.join(conf).join(base[SPLIT])
        save(labels, Path(args.output_dir) / f'labels_1_{name}.csv', inputs,
             f'Upstream L{name} composition; raw gold-row predictions retained')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    competition = DATA / 'rsna-knee-abnormality-detection'
    p = sub.add_parser('base')
    p.add_argument('--competition', default=competition)
    p.set_defaults(function=prepare_base)
    p = sub.add_parser('oof')
    p.add_argument('--config', choices=['cfg_1_85', 'cfg_8_29'], required=True)
    p.add_argument('--run-root')
    p.add_argument('--output')
    p.set_defaults(function=collect_oof)
    p = sub.add_parser('fuse')
    p.add_argument('--source', choices=['cfg_1_85', 'cfg_8_29'], required=True)
    p.add_argument('--oof')
    p.add_argument('--competition', default=competition)
    p.set_defaults(function=fuse)
    p = sub.add_parser('compose')
    for field in ('opus012', 'opus016', 'sol-review', 'gemini-report', 'sol-review-six', 'gemini-images'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--v700')
    p.add_argument('--oa')
    p.add_argument('--output-dir', default=ROOT / 'outputs/labeling/composed')
    p.set_defaults(function=compose)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
