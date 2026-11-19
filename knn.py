import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import compare_distances as cd
import distance as ds
import features as ft
import normalise_features as nf
import query as qr
from preprocessing import out_dir, save_stats

eps = 0.1
candidates = 3
check_csv = os.path.join(out_dir, "knn_check.csv")
weights_csv = os.path.join(out_dir, "knn_l2_weights.csv")


class Engine:
    def __init__(self, normalisation=qr.normalisation, globals_weight=qr.globals_weight, combine="l2"):
        if qr.histogram_metric != "euclidean":
            raise ValueError("kd-tree only works with euclidean histograms")
        self.combine = combine
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
        self.examined = 0
        start = time.perf_counter()
        self.tree = cKDTree(self.vectors)
        print(f"kd-tree ({combine}): {len(self.vectors)} shapes, {self.vectors.shape[1]} dims, "
              f"{1000 * (time.perf_counter() - start):.1f} ms")

    def embed(self, table):
        total = sum(self.weights.values())
        if self.combine == "l2":
            # the tree squares every coordinate difference so block k is scaled by sqrt(w_k)
            factors = {name: np.sqrt(self.weights[name] / total) / self.scales[name] for name in ds.blocks}
        else:
            factors = {name: self.weights[name] / (self.scales[name] * total) for name in ds.blocks}
        return np.hstack([ds.block_matrix(table, name) * factors[name] for name in ds.blocks])

    def exact(self, q, rows):
        diff = self.vectors[rows] - q
        if self.combine == "l2":
            return np.linalg.norm(diff, axis=1)
        return sum(np.linalg.norm(part, axis=1) for part in np.split(diff, self.cuts, axis=1))

    def range(self, q, r, skip=None):
        if self.combine == "l2":
            rows = np.array(self.tree.query_ball_point(q, r), dtype=int)
        else:
            rows = np.array(self.tree.query_ball_point(q, r * (1 + eps), eps=eps), dtype=int)
        if skip is not None:
            rows = rows[rows != skip]
        d = self.exact(q, rows)
        self.examined = len(rows)
        keep = d <= r
        order = np.argsort(d[keep])
        return rows[keep][order], d[keep][order]

    def knn(self, q, k, skip=None):
        if self.combine == "l2":
            d, rows = self.tree.query(q, min(k + (skip is not None), len(self.vectors)))
            d, rows = np.atleast_1d(d), np.atleast_1d(rows)
            self.examined = len(rows)
            if skip is not None:
                d, rows = d[rows != skip], rows[rows != skip]
            return rows[:k], d[:k]
        n = min(k * candidates + 1, len(self.vectors))
        _, rows = self.tree.query(q, n, eps=eps)
        rows = np.atleast_1d(rows)
        if skip is not None:
            rows = rows[rows != skip]
        kth = np.sort(self.exact(q, rows))[min(k, len(rows)) - 1]
        first = len(rows)
        rows, d = self.range(q, kth, skip)
        self.examined += first
        return rows[:k], d[:k]


def main(target, k=qr.default_k, r=None, gui=True, combine="l2"):
    engine = Engine(combine=combine)
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
    suffix = (f"_knn{k}" if r is None else f"_r{r}") + ("_sum" if combine == "sum" else "")
    qr.show(query_path, matches, label, external, gui, suffix)
    return matches


def timed(search, queries):
    start = time.perf_counter()
    for i in queries:
        search(i)
    return 1000 * (time.perf_counter() - start) / len(queries)


def r_precision(matrix, labels):
    sizes = pd.Series(labels).value_counts()
    order = np.argsort(matrix, axis=1, kind="stable")
    return np.mean([(labels[order[i, :sizes[c] - 1]] == c).mean() for i, c in enumerate(labels)])


def check(k=10):
    rows = []
    for combine in ("l2", "sum"):
        engine = Engine(combine=combine)
        n = len(engine.vectors)
        queries = range(n)
        everyone = slice(None)
        labels = engine.database["class"].to_numpy()
        brute = ds.distances_to(engine.table, engine.table, engine.scales, engine.weights, combine=combine)
        gap = max(np.abs(engine.exact(engine.vectors[i], everyone) - brute[i]).max() for i in queries)
        print(f"{combine}: largest difference with distance.py {gap:.1e}")
        np.fill_diagonal(brute, np.inf)

        recall, tied, examined = [], [], []
        for i in queries:
            found, _ = engine.knn(engine.vectors[i], k, skip=i)
            examined.append(engine.examined)
            truth = np.argpartition(brute[i], k)[:k]
            recall.append(len(set(found) & set(truth)) / k)
            # a shape at exactly the distance of the true kth neighbour is an equally good answer!!
            tied.append(np.mean(brute[i, found] <= brute[i, truth].max() + 1e-12))

        def by_tree(i):
            return engine.knn(engine.vectors[i], k, skip=i)

        def by_vectors(i):
            d = engine.exact(engine.vectors[i], everyone)
            d[i] = np.inf
            return np.argpartition(d, k)[:k]

        def like_step4(i):
            d = ds.distances_to(engine.table.iloc[[i]], engine.table, engine.scales, engine.weights,
                                combine=combine)[0]
            d[i] = np.inf
            return np.argsort(d)[:k]

        row = {"search": f"kd-tree, {combine}", "max difference with distance.py": gap,
               f"recall@{k}": np.mean(recall), f"recall@{k} (ties counted)": np.mean(tied),
               "examined median": np.median(examined), "examined 90th pct": np.percentile(examined, 90),
               "tree ms": timed(by_tree, queries), "brute force ms (vectors)": timed(by_vectors, queries),
               "brute force ms (step 4 code)": timed(like_step4, queries),
               f"precision@{k}": cd.precision_at_k(brute.copy(), labels, k),
               "r_precision": r_precision(brute, labels)}

        if combine == "l2":
            # the fastest brute force we found - all squared distances as one matrix-vector product
            squares = (engine.vectors ** 2).sum(axis=1)
            row["brute force ms (matrix-vector)"] = timed(
                lambda i: np.argpartition(squares - 2 * engine.vectors @ engine.vectors[i], k + 1)[:k + 1], queries)
        rows.append(row)
        if combine == "l2":
            approximate = [engine.tree.query(engine.vectors[i], k + 1, eps=eps)[1] for i in queries]
            rows.append({"search": f"kd-tree, {combine}, eps = {eps}",
                         f"recall@{k}": np.mean([len(set(a[a != i][:k]) & set(np.argpartition(brute[i], k)[:k])) / k
                                                 for i, a in enumerate(approximate)]),
                         "tree ms": timed(lambda i: engine.tree.query(engine.vectors[i], k + 1, eps=eps), queries)})

    table = pd.DataFrame(rows)
    save_stats(table, check_csv)
    print(f"\nevery database shape as query, {k} nearest neighbours")
    print(table.round(4).T.to_string(header=False))

    # the weight of the globals for the l2 combination chosen on the tune half as in step 4
    labels = engine.database["class"].to_numpy()
    tune = cd.tune_mask(labels)
    sweep = []
    for w in (1, 2, 3, 4, 6, 10):
        weights = {name: (float(w) if name == "globals" else 1.0) for name in ds.blocks}
        matrix = ds.distances_to(engine.table, engine.table, engine.scales, weights, combine="l2")
        sweep.append({"globals weight": w, "all": cd.precision_at_k(matrix.copy(), labels, k),
                      "tune": cd.precision_at_k(matrix.copy(), labels, k, np.flatnonzero(tune)),
                      "test": cd.precision_at_k(matrix.copy(), labels, k, np.flatnonzero(~tune))})
    sweep = pd.DataFrame(sweep)
    save_stats(sweep, weights_csv)
    print(f"\nl2 combination, precision@{k} for different weights of the globals")
    print(sweep.round(4).to_string(index=False))


if __name__ == "__main__":
    argv = sys.argv[1:]
    usage = ("usage: python knn.py <Class/shape | mesh file> [-k 5 | -r 0.45] [-sum] [-nogui]\n"
             "       python knn.py -check")
    if argv and argv[0] == "-check":
        check()
    elif not argv or argv[0].startswith("-"):
        sys.exit(usage)
    else:
        main(argv[0], qr.flag_value(argv, "-k", qr.default_k, int),
             qr.flag_value(argv, "-r", None, float), gui="-nogui" not in argv,
             combine="sum" if "-sum" in argv else "l2")
