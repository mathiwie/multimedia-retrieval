import os
import random
import numpy as np
import pandas as pd
import open3d as o3d

import distance as ds
import features as ft
from compare_distances import feature_scores
from preprocessing import out_dir, normalised_dir, stats_csv, walk_shapes, save_stats, target_min, target_max

shapes_used = 150
results_csv = os.path.join(out_dir, "d1_surface.csv")
retrieval_csv = os.path.join(out_dir, "d1_surface_retrieval.csv")


def d1_histogram(distances):
    lo, hi = ft.ranges["D1"]
    counts, _ = np.histogram(np.clip(distances, lo, hi), bins=ft.bins, range=(lo, hi))
    return counts / counts.sum()


def surface_points(verts, tris, n, rng):
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    face = rng.choice(len(tris), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return a[face] + r1[:, None] * (b[face] - a[face]) + r2[:, None] * (c[face] - a[face])


def main():
    random.seed(0)
    picked = random.sample(list(walk_shapes(normalised_dir)), shapes_used)
    before = pd.read_csv(stats_csv)
    before["key"] = before["class"] + "/" + before["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    original = dict(zip(before["key"], before["num_verts"]))
    rng = np.random.default_rng(ft.seed)
    rows, vertex_hists = [], []
    for path, cls in picked:
        mesh = o3d.io.read_triangle_mesh(path)
        verts, tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        center = verts.mean(axis=0)
        from_vertices = d1_histogram(np.linalg.norm(verts - center, axis=1))
        from_surface = d1_histogram(np.linalg.norm(surface_points(verts, tris, ft.samples, rng) - center, axis=1))
        n = original[f"{cls}/{os.path.splitext(os.path.basename(path))[0]}"]
        kind = "refined" if n < target_min else "simplified" if n > target_max else "kept"
        rows.append({"path": path, "class": cls, "resampling": kind, "vertices": len(verts),
                     "l1 vertices vs surface": np.abs(from_vertices - from_surface).sum()})
        vertex_hists.append(from_vertices)
    data = pd.DataFrame(rows)
    h = np.array(vertex_hists)
    between = np.abs(h[:, None, :] - h[None, :, :]).sum(-1)[~np.eye(len(h), dtype=bool)].mean()
    data["ratio to signal"] = data["l1 vertices vs surface"] / between
    save_stats(data, results_csv)
    print(f"{shapes_used} random shapes; D1 from all vertices vs D1 from {ft.samples:,} random surface points")
    print(f"signal (L1 between the vertex D1 histograms of two different shapes): {between:.3f}")
    summary = data.groupby("resampling")["l1 vertices vs surface"].describe()[["count", "50%", "75%", "max"]]
    summary["median / signal"] = summary["50%"] / between
    print(summary.round(3).to_string())
    print(f"all shapes: median L1 {data['l1 vertices vs surface'].median():.3f} "
          f"({data['ratio to signal'].median():.1%} of the signal), 90th pct {data['l1 vertices vs surface'].quantile(0.9):.3f}")

    features = pd.read_csv(ft.features_csv)
    surface = features.copy()
    rng = np.random.default_rng(ft.seed)
    for i, path in enumerate(features["path"]):
        mesh = o3d.io.read_triangle_mesh(path)
        verts, tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        hist = d1_histogram(np.linalg.norm(surface_points(verts, tris, ft.samples, rng) - verts.mean(axis=0), axis=1))
        surface.loc[i, [f"D1_{b}" for b in range(ft.bins)]] = hist
    weights = {name: (6.0 if name == "globals" else 1.0) for name in ds.blocks}
    table = pd.DataFrame([{"D1 from": "vertices (used)", **feature_scores(features, weights)},
                          {"D1 from": f"{ft.samples:,} surface points", **feature_scores(surface, weights)}])
    save_stats(table, retrieval_csv)
    print("\nretrieval with each D1 (log normalisation, euclidean, globals x6)")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
