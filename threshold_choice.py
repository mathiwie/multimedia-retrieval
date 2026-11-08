import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import distance as ds
import query as qr
from preprocessing import out_dir

best = {"normalisation": "log", "histogram metric": "euclidean", "globals weight": qr.globals_weight}


def main():
    database = pd.read_csv(os.path.join(out_dir, f"features_{best['normalisation']}.csv"))
    weights = {name: (best["globals weight"] if name == "globals" else 1.0) for name in ds.blocks}
    scales = ds.fit_scales(database, best["histogram metric"])
    matrix = ds.distances_to(database, database, scales, weights, best["histogram metric"])
    labels = database["class"].to_numpy()
    np.fill_diagonal(matrix, np.nan)

    same = labels[:, None] == labels[None, :]
    same_class = matrix[same & ~np.isnan(matrix)]
    other_class = matrix[~same & ~np.isnan(matrix)]
    print(f"same class:      median {np.median(same_class):.3f}  90th {np.percentile(same_class, 90):.3f}")
    print(f"different class: median {np.median(other_class):.3f}  10th {np.percentile(other_class, 10):.3f}")

    ks = list(range(1, 21))
    precision = []
    for k in ks:
        nearest = np.argsort(np.nan_to_num(matrix, nan=np.inf), axis=1)[:, :k]
        precision.append((labels[nearest] == labels[:, None]).mean())

    thresholds = np.linspace(0.2, 1.2, 21)
    returned, correct = [], []
    for t in thresholds:
        hits = matrix < t
        counts = hits.sum(axis=1)
        returned.append(counts.mean())
        with np.errstate(invalid="ignore"):
            share = np.where(counts > 0, (hits & same).sum(axis=1) / np.maximum(counts, 1), np.nan)
        correct.append(np.nanmean(share))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].hist(other_class, bins=60, density=True, alpha=0.6, label="different class", color="indianred")
    ax[0].hist(same_class, bins=60, density=True, alpha=0.6, label="same class", color="seagreen")
    ax[0].set_title("distance distributions")
    ax[0].set_xlabel("distance")
    ax[0].legend()

    ax[1].plot(ks, precision, marker="o")
    ax[1].set_title("precision@K")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("share of matches in the query's class")

    ax[2].plot(thresholds, returned, marker="o", color="steelblue", label="shapes returned (avg)")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("threshold t")
    ax[2].set_ylabel("shapes returned per query")
    twin = ax[2].twinx()
    twin.plot(thresholds, correct, marker="s", color="seagreen", label="share in query's class")
    twin.set_ylabel("share in query's class")
    ax[2].set_title("what a threshold returns")
    lines = ax[2].get_lines() + twin.get_lines()
    ax[2].legend(lines, [line.get_label() for line in lines], loc="center right")

    plt.tight_layout()
    fname = os.path.join(out_dir, "threshold_choice.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved", fname)
    table = pd.DataFrame({"threshold": thresholds, "avg shapes returned": returned,
                          "share in query class": correct})
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
