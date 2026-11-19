import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import open3d as o3d

import distance as ds
import features as ft
import query as qr
from compare_distances import feature_scores
from preprocessing import out_dir, save_stats

# the curvature histogram at another patch radius, for every shape, and
# retrieval with each radius
other_radius = 0.08
other_csv = os.path.join(out_dir, f"curvature_r{other_radius}.csv")
results_csv = os.path.join(out_dir, "curvature_radius_check.csv")
weights = {name: (qr.globals_weight if name == "globals" else 1.0) for name in ds.blocks}


def curvature_row(path, radius=other_radius):
    values, area, dropped = ft.curvature_values(o3d.io.read_triangle_mesh(path), radius=radius)
    row = {"path": path, "curvature_dropped": dropped}
    if len(values):
        row.update(ft.histogram_features({"CURV": values}, {"CURV": area}))
    return row


def main():
    features = pd.read_csv(ft.features_csv)
    if os.path.isfile(other_csv) and "-redo" not in sys.argv[1:]:
        print("loading", other_csv)
        other = pd.read_csv(other_csv)
    else:
        print(f"curvature at r = {other_radius} for {len(features)} shapes...")
        with Pool(os.cpu_count()) as pool:
            other = pd.DataFrame(pool.map(curvature_row, features["path"], chunksize=4))
        save_stats(other, other_csv)
    columns = [f"CURV_{i}" for i in range(ft.n_bins("CURV"))]
    swapped = features.copy()
    swapped[columns] = other.set_index("path").loc[features["path"], columns].to_numpy()
    table = pd.DataFrame([{"radius": ft.curvature_radius, **feature_scores(features, weights)},
                          {"radius": other_radius, **feature_scores(swapped, weights)}])
    save_stats(table, results_csv)
    print(f"retrieval with the curvature histogram at each radius (log, euclidean, globals x{qr.globals_weight:g})")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
