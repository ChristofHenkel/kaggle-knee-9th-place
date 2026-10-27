import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .artifacts import sha256
from .prepare_knee_labels import TARGETS, canonical, table


def fit_mapping(features, targets, validation, cfg):
    rows, columns = features.shape
    outputs = targets.shape[1]
    initial = np.zeros((columns + 1, outputs), dtype=np.float64)
    initial[-1] = logit(targets.mean(0).clip(1e-4, 1 - 1e-4))

    def objective(flat):
        weight = flat.reshape(columns + 1, outputs)
        scores = features @ weight[:-1] + weight[-1]
        residual = (expit(scores) - targets) / rows
        loss = (np.logaddexp(0, scores) - targets * scores).sum() / rows
        loss += cfg.mapping_logistic_l2 * np.square(weight[:-1]).sum() / 2
        gradient = np.vstack([features.T @ residual + cfg.mapping_logistic_l2 * weight[:-1],
                              residual.sum(0)])
        return float(loss), gradient.ravel()

    result = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
                      options={'maxiter': cfg.mapping_logistic_iterations, 'ftol': 1e-9, 'gtol': 1e-6})
    weight = result.x.reshape(columns + 1, outputs)
    return validation @ weight[:-1] + weight[-1], {'weight': weight[:-1], 'bias': weight[-1]}


def map_probabilities(probabilities, frame, labels, output, iterations=500, threads=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    split = canonical().loc[labels.index]
    indices = pd.Index(labels.index).get_indexer(frame.StudyInstanceUID)
    if (indices < 0).any() or len(np.unique(indices)) != len(labels):
        raise ValueError('Categorical predictions do not cover every competition study')
    pooled = np.zeros((len(labels), probabilities.shape[1]), dtype=np.float64)
    np.add.at(pooled, indices, probabilities)
    pooled /= np.bincount(indices, minlength=len(labels))[:, None]
    gold = split.is_gold.to_numpy(bool)
    gold_groups = set(split.loc[gold, 'fold_group'])
    eligible = ~gold & ~split.fold_group.isin(gold_groups).to_numpy()
    targets = labels[TARGETS].to_numpy(np.float64)
    series_scores = np.empty((len(frame), len(TARGETS)), dtype=np.float64)
    cfg = SimpleNamespace(mapping_logistic_l2=.01, mapping_logistic_iterations=iterations)
    with threadpool_limits(limits=threads):
        for fold in range(5):
            training = eligible & split.fold.ne(fold).to_numpy()
            held_out = split.fold.eq(fold).to_numpy()
            scaler = StandardScaler().fit(pooled[training])
            _, state = fit_mapping(scaler.transform(pooled[training]), targets[training],
                                   scaler.transform(pooled[held_out]), cfg)
            state.update(mean=scaler.mean_, scale=scaler.scale_)
            np.savez_compressed(output / f'fold{fold}.npz', **state)
            series = held_out[indices]
            series_scores[series] = ((probabilities[series] - state['mean']) / state['scale']) @ state['weight'] + state['bias']
    mapped = np.zeros_like(targets)
    np.add.at(mapped, indices, expit(series_scores))
    mapped /= np.bincount(indices, minlength=len(labels))[:, None]
    result = pd.DataFrame(mapped, index=labels.index, columns=TARGETS)
    result.to_csv(output / 'oof.csv')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    labels = table(args.labels).loc[canonical().index]
    with np.load(args.predictions) as data:
        probabilities = data['probabilities']
        frame = pd.DataFrame({'StudyInstanceUID': data['StudyInstanceUID']})
    map_probabilities(probabilities, frame, labels, args.output, threads=args.threads)
    provenance = {'recipe': 'categorical; standardize study means; five soft-logistic fold maps; map series then mean probabilities; no PCA',
                  'gold_policy': 'exclude gold studies and report groups shared with gold from fitting',
                  'sources': {args.predictions: sha256(args.predictions), args.labels: sha256(args.labels)}}
    (Path(args.output) / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
