import os
import numpy as np
import pandas as pd

import features as ft
from preprocessing import out_dir, save_stats

globals_used = ["area", "compactness", "rectangularity", "diameter", "convexity", "eccentricity"]
skewed = ["area", "compactness", "eccentricity"]
holdout_csv = os.path.join(out_dir, "holdout.csv")
holdout_size = 10


def histogram_columns(data):
    return [c for c in data.columns if c.split("_")[0] in ft.ranges and c.split("_")[-1].isdigit()]


def split_holdout(data, n=holdout_size, seed=0):
    rng = np.random.default_rng(seed)
    classes = sorted(data["class"].unique())
    picked = []
    for cls in rng.choice(classes, size=n, replace=False):
        rows = data[data["class"] == cls]
        picked.append(rows.index[rng.integers(0, len(rows))])
    return data.drop(index=picked), data.loc[picked]


def fit(database, kind="log"):
    values = database[globals_used].astype(float).copy()
    if kind == "log":
        values[skewed] = np.log(values[skewed])
    if kind == "minmax":
        return {"kind": kind, "lo": values.min(), "range": (values.max() - values.min()).replace(0, 1)}
    return {"kind": kind, "mean": values.mean(), "std": values.std().replace(0, 1)}


def apply(data, params):
    values = data[globals_used].astype(float).copy()
    if params["kind"] == "log":
        values[skewed] = np.log(values[skewed])
    if params["kind"] == "minmax":
        return (values - params["lo"]) / params["range"]
    return (values - params["mean"]) / params["std"]


def normalised_table(data, params):
    return pd.concat([data[["path", "class"]], apply(data, params),
                      data[histogram_columns(data)]], axis=1)


def main():
    data = pd.read_csv(ft.features_csv)
    database, holdout = split_holdout(data)
    save_stats(holdout[["path", "class"] + globals_used], holdout_csv)
    print(f"database {len(database)} shapes, held out {len(holdout)}")
    for kind in ("minmax", "standard", "log"):
        params = fit(database, kind)
        table = normalised_table(database, params)
        save_stats(table, os.path.join(out_dir, f"features_{kind}.csv"))
        print(f"  {kind}: global descriptor ranges after normalising")
        print(table[globals_used].describe().T[["mean", "std", "min", "max"]].round(2).to_string())


def print_normalised(key, kind="log"):
    data = pd.read_csv(ft.features_csv)
    database, _ = split_holdout(data)
    table = normalised_table(data, fit(database, kind))
    path = os.path.join(ft.normalised_dir, key + ".obj").replace("\\", "/")
    match = table[table["path"].str.replace("\\", "/", regex=False) == path]
    if len(match) == 0:
        raise SystemExit(f"no features for {key} in {ft.features_csv}")
    held_out = match.index[0] not in database.index
    print(f"{key} ({kind} normalisation{', held out of the database' if held_out else ''})")
    print(match.iloc[0][globals_used].astype(float).round(4).to_string())
    for name in ft.ranges:
        values = match.iloc[0][[f"{name}_{i}" for i in range(ft.n_bins(name))]].astype(float)
        print(f"{name:3s} " + " ".join(f"{v:.3f}" for v in values))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_normalised(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "log")
    else:
        main()
