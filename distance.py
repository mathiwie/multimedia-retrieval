import numpy as np
from scipy.spatial.distance import cdist

import features as ft
import normalise_features as nf

blocks = ["globals"] + list(ft.ranges)


def block_columns(data, name):
    if name == "globals":
        return nf.globals_used
    return [c for c in data.columns if c.startswith(name + "_") and c.split("_")[-1].isdigit()]


def block_matrix(data, name):
    return data[block_columns(data, name)].to_numpy(dtype=float)


def block_distances(query, database, name, histogram_metric="euclidean"):
    if name == "globals":
        return cdist(query, database, "euclidean")
    if histogram_metric == "cosine":
        return cdist(query, database, "cosine")
    if histogram_metric == "emd":
        return cdist(np.cumsum(query, axis=1), np.cumsum(database, axis=1), "cityblock")
    return cdist(query, database, "euclidean")


def fit_scales(database, histogram_metric="euclidean", sample=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(database), size=min(sample, len(database)), replace=False)
    scales = {}
    for name in blocks:
        matrix = block_matrix(database, name)
        distances = block_distances(matrix[rows], matrix[rows], name, histogram_metric)
        mean = distances[~np.eye(len(rows), dtype=bool)].mean()
        scales[name] = mean if mean > 0 else 1.0
    return scales


def distances_to(query, database, scales, weights=None, histogram_metric="euclidean"):
    weights = weights or {name: 1.0 for name in blocks}
    total = np.zeros((len(query), len(database)))
    for name in blocks:
        if not weights.get(name):
            continue
        part = block_distances(block_matrix(query, name), block_matrix(database, name),
                               name, histogram_metric)
        total += weights[name] * part / scales[name]
    return total / sum(weights.get(name, 0) for name in blocks)
