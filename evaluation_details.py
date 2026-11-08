import os
import numpy as np
import pandas as pd

import distance as ds
import features as ft
import query as qr
from preprocessing import out_dir

best = {"normalisation": "log", "histogram metric": "euclidean", "globals weight": qr.globals_weight}
k = 10
duplicate_distance = 0.02
shown = ["Bicycle", "ComputerKeyboard", "HumanHead", "Apartment", "FloorLamp", "Vase", "Drum",
         "Rocket", "Starship", "Sign", "Car"]


def main():
    database = pd.read_csv(os.path.join(out_dir, f"features_{best['normalisation']}.csv"))
    weights = {name: (best["globals weight"] if name == "globals" else 1.0) for name in ds.blocks}
    scales = ds.fit_scales(database, best["histogram metric"])
    matrix = ds.distances_to(database, database, scales, weights, best["histogram metric"])
    np.fill_diagonal(matrix, np.inf)
    labels = database["class"].to_numpy()
    order = np.argsort(matrix, axis=1)
    sizes = pd.Series(labels).value_counts()

    same = labels[order] == labels[:, None]
    p10 = same[:, :k].mean(axis=1)
    r = np.array([same[i, :sizes[labels[i]] - 1].mean() for i in range(len(labels))])
    print(f"{len(labels)} queries: precision@{k} {p10.mean():.3f}, R-precision {r.mean():.3f}")
    print(f"all {k} results correct for {(p10 == 1).mean():.1%} of the queries, none for {(p10 == 0).mean():.1%}")

    classes = pd.DataFrame({"class": labels, "r": r}).groupby("class")["r"].mean()
    print(f"classes with R-precision above 0.3: {(classes > 0.3).sum()}, 0.1 to 0.3: "
          f"{((classes >= 0.1) & (classes <= 0.3)).sum()}, below 0.1: {(classes < 0.1).sum()}")
    ranks = pd.DataFrame({"size": sizes[classes.index].to_numpy(), "r": classes.to_numpy()}).rank()
    print(f"Spearman rank correlation of class size and R-precision: {ranks['size'].corr(ranks['r']):.2f}")

    hits = matrix < 0.35
    counts = hits.sum(axis=1)
    print(f"\nthreshold 0.35: {(counts == 0).mean():.0%} of the queries get nothing, median {np.median(counts):.0f}, "
          f"90th percentile {np.percentile(counts, 90):.0f}, most {counts.max()}")

    twin = matrix.min(axis=1) < duplicate_distance
    print(f"\n{twin.sum()} shapes have another shape closer than {duplicate_distance}")
    print(f"without them as queries: precision@{k} {p10[~twin].mean():.3f}, R-precision {r[~twin].mean():.3f}")

    raw = pd.read_csv(ft.features_csv).set_index("path").loc[database["path"]]
    flagged = (raw["hull_fallback"].astype(bool) | raw["curvature_doubtful"].astype(bool)).to_numpy()
    print(f"\n{flagged.sum()} shapes with a hull fallback or a doubtful curvature: "
          f"R-precision {r[flagged].mean():.3f} against {r[~flagged].mean():.3f}, "
          f"precision@{k} {p10[flagged].mean():.3f} against {p10[~flagged].mean():.3f}")

    print(f"\nclasses of the wrong results among the {k} nearest neighbours")
    for name in shown:
        found = labels[order[labels == name, :k]].ravel()
        wrong = pd.Series(found[found != name]).value_counts(normalize=True).head(4)
        print(f"  {name:18s} " + ", ".join(f"{c} {v:.0%}" for c, v in wrong.items()))


if __name__ == "__main__":
    main()
