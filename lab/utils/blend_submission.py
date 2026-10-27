import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare_knee_labels import TARGETS


COMPONENTS = ['v313', 'v311', '1_91', '8_31', '8_32', 'l185conf1']
WEIGHTS = [.2, .2, .4 / 3, .4 / 3, .4 / 3, .2]


def blend(paths, sample_path, output):
    sample = pd.read_csv(sample_path).set_index('StudyInstanceUID', verify_integrity=True)
    predictions = []
    for name, path in zip(COMPONENTS, paths):
        frame = pd.read_csv(path).set_index('StudyInstanceUID', verify_integrity=True)
        if set(frame.index) != set(sample.index) or frame.columns.tolist() != TARGETS:
            raise ValueError(f'Study coverage or targets differ: {name}')
        values = frame.loc[sample.index, TARGETS].to_numpy(np.float64)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError(f'Invalid probabilities: {name}')
        predictions.append(values)
    result = pd.DataFrame(np.average(np.stack(predictions), axis=0, weights=WEIGHTS),
                          index=sample.index, columns=TARGETS)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--components', nargs=6, required=True, metavar='CSV', help=', '.join(COMPONENTS))
    parser.add_argument('--sample-submission', required=True)
    parser.add_argument('--output', default='submission.csv')
    args = parser.parse_args()
    blend(args.components, args.sample_submission, args.output)


if __name__ == '__main__':
    main()
