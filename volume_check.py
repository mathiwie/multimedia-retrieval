import os
import random
import time
import numpy as np
import pandas as pd
import open3d as o3d

import features as ft
from preprocessing import out_dir, normalised_dir, walk_shapes, save_stats

shapes_used = 40
resolutions = [32, 64, 128, 256]
ray_counts = [1, 3, 5]
check_csv = os.path.join(out_dir, "volume_check.csv")


def main():
    random.seed(0)
    picked = random.sample(list(walk_shapes(normalised_dir)), shapes_used)
    rows = []
    for path, cls in picked:
        mesh = o3d.io.read_triangle_mesh(path)
        hull, _ = mesh.compute_convex_hull()
        row = {"path": path, "class": cls, "hull_volume": ft.closed_mesh_volume(hull),
               "watertight": mesh.is_watertight()}
        for res in resolutions:
            start = time.perf_counter()
            row[f"volume_{res}"] = ft.mesh_volume(mesh, res)
            row[f"seconds_{res}"] = time.perf_counter() - start
        for rays in ray_counts[1:]:
            row[f"volume_{ft.volume_res}_rays{rays}"] = ft.mesh_volume(mesh, ft.volume_res, rays)
        start = time.perf_counter()
        ft.global_features(mesh)
        row["seconds_global_features"] = time.perf_counter() - start
        rows.append(row)
    data = pd.DataFrame(rows)
    save_stats(data, check_csv)

    finest = data[f"volume_{resolutions[-1]}"]
    print(f"{shapes_used} random shapes, {data['watertight'].sum()} of them watertight")
    print(f"\n{'grid':>6s}  {'cells along longest side':>24s}  {'median |V - V_256| / V_256':>27s}  {'90th pct':>9s}  {'s per shape':>11s}")
    for res in resolutions:
        rel = (data[f"volume_{res}"] - finest).abs() / finest
        print(f"{res:>6d}  {res:>24d}  {rel.median():>27.2%}  {rel.quantile(0.9):>9.2%}  {data[f'seconds_{res}'].mean():>11.2f}")

    print(f"\ninside test with several rays (majority vote), grid {ft.volume_res}:")
    base = data[f"volume_{ft.volume_res}"]
    print(f"  1 ray : volume above hull volume for {(base > data['hull_volume']).sum()} shapes")
    for rays in ray_counts[1:]:
        v = data[f"volume_{ft.volume_res}_rays{rays}"]
        rel = (v - base).abs() / base
        print(f"  {rays} rays: volume above hull volume for {(v > data['hull_volume']).sum()} shapes, "
              f"median change vs 1 ray {rel.median():.2%}, 90th pct {rel.quantile(0.9):.2%}")
    print(f"\nall six global descriptors together: {data['seconds_global_features'].mean():.2f} s per shape on average")


if __name__ == "__main__":
    main()
