import numpy as np


def pick_centres(n_slices, depth, training):
    if not training:
        return np.round(np.linspace(0, n_slices - 1, depth)).astype(np.int64)
    slack = n_slices - depth
    return np.arange(depth) + np.sort(np.random.randint(0, slack + 1, size=depth))
