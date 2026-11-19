import os
import random
import time
from math import comb
import numpy as np
import pandas as pd
import open3d as o3d

import features as ft
from preprocessing import out_dir, normalised_dir, walk_shapes, save_stats

shapes_used = 40
resolutions = [32, 64, 128, 256]
ray_counts = [1, 3, 5, 7, 9]
reference_rays = 25
check_csv = os.path.join(out_dir, "volume_check.csv")


def main():
    random.seed(0)
    picked = random.sample(list(walk_shapes(normalised_dir)), shapes_used)
    rows = []
    for path, cls in picked:
        mesh = o3d.io.read_triangle_mesh(path)
        hull, _ = mesh.compute_convex_hull()
        row = {"path": path, "class": cls,
               "hull_volume": ft.closed_mesh_volume(hull),
               "watertight": mesh.is_watertight()}
        for res in resolutions:
            start = time.perf_counter()
            row[f"volume_{res}"], row[f"leakage_{res}"
                                      ] = ft.mesh_volume(mesh, res)
            row[f"seconds_{res}"] = time.perf_counter() - start
        for rays in ray_counts + [reference_rays]:
            start = time.perf_counter()
            row[f"volume_rays{rays}"], _ = ft.mesh_volume(
                mesh, ft.volume_res, rays)
            row[f"seconds_rays{rays}"] = time.perf_counter() - start
        start = time.perf_counter()
        ft.global_features(mesh)
        row["seconds_global_features"] = time.perf_counter() - start
        rows.append(row)
    data = pd.DataFrame(rows)
    save_stats(data, check_csv)

    finest = data[f"volume_{resolutions[-1]}"]
    print(f"{shapes_used} random shapes, {data['watertight'].sum()} of",
          "them watertight")
    print(f"\n{'grid':>6s}  {'median |V - V_256| / V_256':>27s}  {'90th pct':>9s}  {'s per shape':>11s}")
    for res in resolutions:
        rel = (data[f"volume_{res}"] - finest).abs() / finest
        print(f"{res:>6d}  {rel.median():>27.2%}  {rel.quantile(0.9):>9.2%}  {data[f'seconds_{res}'].mean():>11.2f}")

    directions = ft.ray_directions(ft.volume_rays)
    angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1)))
    np.fill_diagonal(angles, 360)
    print(f"\nthe {ft.volume_rays} ray direction:")
    print(np.round(directions, 3))
    print(f"  smallest angle between two directions {angles.min():.1f} degrees, "
          f"smallest angle to a coordinate axis {np.degrees(np.arccos(np.abs(directions).max(axis=1))).min():.1f} degrees")
    print(f"\nmajority vote over several ray directions, grid {ft.volume_res} (used: {ft.volume_rays} rays), "
          f"compared with a vote over {reference_rays} rays:")
    base = data[f"volume_rays{reference_rays}"]
    for rays in ray_counts + [reference_rays]:
        v = data[f"volume_rays{rays}"]
        rel = (v - base).abs() / base
        print(f"  {rays:>2d} rays: median |V - V_{reference_rays}| / V_{reference_rays} {rel.median():6.2%}, "
              f"90th pct {rel.quantile(0.9):6.2%}, above hull volume for {(v > data['hull_volume']).sum()} shapes, "
              f"{data[f'seconds_rays{rays}'].mean():.2f} s per shape")
    print("  if every ray failed independently with probability p, a majority vote would fail with probability")
    for p in (0.1, 0.2, 0.3):
        fail = [sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(n // 2 + 1, n + 1)) for n in ray_counts]
        print(f"    p = {p}: " + ", ".join(f"{n} rays {f:.3f}" for n, f in zip(ray_counts, fail)))
    print(f"\nall six global descriptors together: {data['seconds_global_features'].mean():.2f} s per shape on average")


if __name__ == "__main__":
    main()
