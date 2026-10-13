import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import distance as ds
import features as ft
import normalise_features as nf
import query as qr

eps = 0.1
candidates = 3


class Engine:
    def __init__(self, normalisation=qr.normalisation, globals_weight=qr.globals_weight):
        if qr.histogram_metric != "euclidean":
            raise ValueError("kd-tree only works with euclidean histograms")
        everything = pd.read_csv(ft.features_csv)
        database, _ = nf.split_holdout(everything)
        self.params = nf.fit(database, normalisation)
        self.database = database.reset_index(drop=True)
        self.database["path"] = self.database["path"].str.replace("\\", "/", regex=False)
        self.paths = self.database["path"].to_numpy()
        self.table = nf.normalised_table(self.database, self.params)
        self.weights = {name: (globals_weight if name == "globals" else 1.0) for name in ds.blocks}
        self.scales = ds.fit_scales(self.table, "euclidean")
        sizes = [len(ds.block_columns(self.table, name)) for name in ds.blocks]
        self.cuts = np.cumsum(sizes)[:-1]
        self.vectors = self.embed(self.table)
        start = time.perf_counter()
        self.tree = cKDTree(self.vectors)
        print(f"kd-tree: {len(self.vectors)} shapes, {self.vectors.shape[1]} dims, "
              f"{1000 * (time.perf_counter() - start):.1f} ms")

    def embed(self, table):
        total = sum(self.weights.values())
        return np.hstack([ds.block_matrix(table, name) * self.weights[name] / (self.scales[name] * total)
                          for name in ds.blocks])

    def exact(self, q, rows):
        diff = self.vectors[rows] - q
        return sum(np.linalg.norm(part, axis=1) for part in np.split(diff, self.cuts, axis=1))

    def range(self, q, r, skip=None):
        rows = np.array(self.tree.query_ball_point(q, r * (1 + eps), eps=eps), dtype=int)
        if skip is not None:
            rows = rows[rows != skip]
        d = self.exact(q, rows)
        keep = d <= r
        order = np.argsort(d[keep])
        return rows[keep][order], d[keep][order]

    def knn(self, q, k, skip=None):
        n = min(k * candidates + 1, len(self.vectors))
        _, rows = self.tree.query(q, n, eps=eps)
        rows = np.atleast_1d(rows)
        if skip is not None:
            rows = rows[rows != skip]
        kth = np.sort(self.exact(q, rows))[min(k, len(rows)) - 1]
        rows, d = self.range(q, kth, skip)
        return rows[:k], d[:k]


def main(target, k=qr.default_k, r=None, gui=True):
    engine = Engine()
    raw, query_path = qr.query_row(target, set(engine.paths))
    q = engine.embed(nf.normalised_table(raw.reset_index(drop=True), engine.params))[0]
    same = np.flatnonzero(engine.paths == query_path.replace("\\", "/"))
    skip = same[0] if len(same) else None

    start = time.perf_counter()
    rows, d = engine.knn(q, k, skip) if r is None else engine.range(q, r, skip)
    took = 1000 * (time.perf_counter() - start)
    search = f"{k}-nn" if r is None else f"r-nn (r = {r})"
    print(f"{search}: {len(rows)} shapes, {took:.2f} ms")

    matches = engine.database.loc[rows, ["path", "class"]].assign(distance=d)
    print(f"\n{'rank':>4}  {'shape':28s} {'class':24s} distance")
    for rank, (path, cls, dist) in enumerate(zip(matches["path"], matches["class"], matches["distance"]), 1):
        print(f"{rank:>4}  {qr.key(path):28s} {cls:24s} {dist:.4f}")
    if len(matches) == 0:
        print("  (nothing within that distance)")
        return matches

    external = qr.os.path.isfile(target) and not target.startswith(qr.normalised_dir)
    label = qr.key(target) if external else qr.key(query_path)
    qr.show(query_path, matches, label, external, gui, f"_knn{k}" if r is None else f"_r{r}")
    return matches


def check(k=10, compared=50):
    engine = Engine()
    n = len(engine.vectors)
    brute = ds.distances_to(engine.table, engine.table, engine.scales, engine.weights)
    gap = max(np.abs(engine.exact(engine.vectors[i], np.arange(n)) - brute[i]).max() for i in range(compared))
    print(f"max difference with distance.py: {gap:.2e}")

    recall, tree_time, brute_time = [], 0.0, 0.0
    everyone = slice(None)
    for i in range(n):
        q = engine.vectors[i]
        start = time.perf_counter()
        rows, _ = engine.knn(q, k, skip=i)
        tree_time += time.perf_counter() - start
        start = time.perf_counter()
        d = engine.exact(q, everyone)
        d[i] = np.inf
        truth = np.argpartition(d, k)[:k]
        brute_time += time.perf_counter() - start
        recall.append(len(set(rows) & set(truth)) / k)
    print(f"{k}-nn recall vs brute force: {np.mean(recall):.4f}")
    print(f"kd-tree: {1000 * tree_time / n:.3f} ms per query")
    print(f"brute force: {1000 * brute_time / n:.3f} ms per query")


if __name__ == "__main__":
    argv = sys.argv[1:]
    usage = "usage: python knn.py <Class/shape | mesh file> [-k 5 | -r 0.4] [-nogui]\n       python knn.py -check"
    if argv and argv[0] == "-check":
        check()
    elif not argv or argv[0].startswith("-"):
        sys.exit(usage)
    else:
        main(argv[0], qr.flag_value(argv, "-k", qr.default_k, int),
             qr.flag_value(argv, "-r", None, float), gui="-nogui" not in argv)