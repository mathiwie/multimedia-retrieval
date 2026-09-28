import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import distance as ds
import normalise_features as nf
from preprocessing import out_dir, save_stats

results_csv = os.path.join(out_dir, "distance_comparison.csv")
ks = (1, 5, 10)


def precision_at_k(matrix, labels, k):
    np.fill_diagonal(matrix, np.inf)
    nearest = np.argpartition(matrix, k, axis=1)[:, :k]
    return (labels[nearest] == labels[:, None]).mean()


def evaluate(database, histogram_metric, weights):
    scales = ds.fit_scales(database, histogram_metric)
    matrix = ds.distances_to(database, database, scales, weights, histogram_metric)
    labels = database["class"].to_numpy()
    return {f"precision@{k}": precision_at_k(matrix, labels, k) for k in ks}


def main():
    rows = []
    tables = {kind: pd.read_csv(os.path.join(out_dir, f"features_{kind}.csv"))
              for kind in ("minmax", "standard", "log")}
    all_blocks = {name: 1.0 for name in ds.blocks}
    weightings = {
        "all blocks": all_blocks,
        "globals only": {"globals": 1.0},
        "histograms only": {name: (0.0 if name == "globals" else 1.0) for name in ds.blocks},
        "no curvature": {name: (0.0 if name == "SI" else 1.0) for name in ds.blocks},
        "globals x3": {name: (3.0 if name == "globals" else 1.0) for name in ds.blocks},
        "globals x6": {name: (6.0 if name == "globals" else 1.0) for name in ds.blocks},
        "globals x10": {name: (10.0 if name == "globals" else 1.0) for name in ds.blocks},
        "globals x20": {name: (20.0 if name == "globals" else 1.0) for name in ds.blocks},
    }

    for kind, table in tables.items():
        for metric in ("euclidean", "cosine", "emd"):
            for weight_name, weights in weightings.items():
                balanced = weight_name in ("all blocks", "globals x6")
                if not balanced and (kind != "log" or metric != "euclidean"):
                    continue
                if balanced and weight_name == "globals x6" and kind != "log":
                    continue
                scores = evaluate(table, metric, weights)
                rows.append({"normalisation": kind, "histogram distance": metric,
                             "blocks": weight_name, **scores})
                print(f"{kind:9s} {metric:10s} {weight_name:16s} " +
                      "  ".join(f"p@{k} {scores[f'precision@{k}']:.3f}" for k in ks))

    comparison = pd.DataFrame(rows).sort_values("precision@10", ascending=False)
    save_stats(comparison, results_csv)
    print("\nbest:", comparison.iloc[0].to_dict())

    main_rows = comparison[comparison["blocks"] == "all blocks"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in main_rows["histogram distance"].unique():
        part = main_rows[main_rows["histogram distance"] == metric]
        ax.bar([f"{r.normalisation}\n{metric}" for r in part.itertuples()],
               part["precision@10"], label=metric)
    ax.set_ylabel("precision@10")
    ax.set_title("normalisation and histogram distance (all blocks)")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, "distance_comparison.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    main()
