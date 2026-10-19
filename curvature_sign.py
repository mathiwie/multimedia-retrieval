import os
import pandas as pd

import distance as ds
import features as ft
from compare_distances import feature_scores
from preprocessing import out_dir, save_stats

results_csv = os.path.join(out_dir, "curvature_sign.csv")
weights = {name: (6.0 if name == "globals" else 1.0) for name in ds.blocks}


def unsigned(features):
    half = ft.bins // 2
    signed = [f"CURV_{i}" for i in range(ft.bins)]
    folded = pd.DataFrame({f"CURV_{j}": features[f"CURV_{half + j}"] + features[f"CURV_{half - 1 - j}"]
                           for j in range(half)})
    return pd.concat([features.drop(columns=signed), folded], axis=1)


def main():
    features = pd.read_csv(ft.features_csv)
    variants = {"signed curvature (used)": (features, weights),
                "unsigned |curvature|": (unsigned(features), weights),
                "no curvature": (features, {**weights, "CURV": 0.0})}
    rows = [{"curvature histogram": name, **feature_scores(data, w)} for name, (data, w) in variants.items()]
    table = pd.DataFrame(rows)
    save_stats(table, results_csv)
    print("log normalisation, euclidean, globals x6; the unsigned histogram folds bin -x onto bin +x (20 bins)")
    print(table.round(3).to_string(index=False))
    signed = features[[f"CURV_{i}" for i in range(ft.bins // 2)]].sum(axis=1)
    print(f"share of the surface with a negative curvature: median {signed.median():.3f}, "
          f"90th pct {signed.quantile(0.9):.3f}")


if __name__ == "__main__":
    main()
