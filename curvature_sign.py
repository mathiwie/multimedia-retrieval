import os
import pandas as pd

import distance as ds
import features as ft
import query as qr
from compare_distances import feature_scores
from preprocessing import out_dir, save_stats

results_csv = os.path.join(out_dir, "curvature_sign.csv")
weights = {name: (qr.globals_weight if name == "globals" else 1.0) for name in ds.blocks}


def unsigned(features):
    n = ft.n_bins("CURV")
    half = n // 2
    signed = [f"CURV_{i}" for i in range(n)]
    if n % 2:
        # odd: the middle bin holds the flat values and stays as it is, bin half + j gets bin half - j
        folded = {"CURV_0": features[f"CURV_{half}"]}
        folded.update({f"CURV_{j}": features[f"CURV_{half + j}"] + features[f"CURV_{half - j}"]
                       for j in range(1, half + 1)})
    else:
        folded = {f"CURV_{j}": features[f"CURV_{half + j}"] + features[f"CURV_{half - 1 - j}"]
                  for j in range(half)}
    return pd.concat([features.drop(columns=signed), pd.DataFrame(folded)], axis=1)


def main():
    features = pd.read_csv(ft.features_csv)
    variants = {"signed curvature (used)": (features, weights),
                "unsigned |curvature|": (unsigned(features), weights),
                "no curvature": (features, {**weights, "CURV": 0.0})}
    rows = [{"curvature histogram": name, **feature_scores(data, w)} for name, (data, w) in variants.items()]
    table = pd.DataFrame(rows)
    save_stats(table, results_csv)
    print(f"log normalisation, euclidean, globals x{qr.globals_weight:g}; the unsigned histogram folds bin -x onto bin +x "
          f"({ft.n_bins('CURV') // 2 + ft.n_bins('CURV') % 2} bins)")
    print(table.round(3).to_string(index=False))
    signed = features[[f"CURV_{i}" for i in range(ft.n_bins("CURV") // 2)]].sum(axis=1)
    print(f"share of the surface with a negative curvature: median {signed.median():.3f}, "
          f"90th pct {signed.quantile(0.9):.3f}")


if __name__ == "__main__":
    main()
