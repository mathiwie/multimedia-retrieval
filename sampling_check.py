import os
import random
import numpy as np
import pandas as pd
import open3d as o3d

import features as ft
from preprocessing import out_dir, normalised_dir, walk_shapes, save_stats

shapes_used = 25
names = ["A3", "D1", "D2", "D3", "D4"]
check_csv = os.path.join(out_dir, "sampling_check.csv")


def vertex_values(mesh, rng):
    # the first version: random vertices instead of random points on the surface, D1 from every vertex
    verts = np.asarray(mesh.vertices)
    pick = lambda k: verts[rng.integers(0, len(verts), size=(ft.samples, k))]
    a3 = pick(3)
    u, v = a3[:, 1] - a3[:, 0], a3[:, 2] - a3[:, 0]
    lengths = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cosines = np.divide(np.einsum("ij,ij->i", u, v), lengths, out=np.zeros(len(u)), where=lengths > 0)
    d2, d3, d4 = pick(2), pick(3), pick(4)
    edges = d4[:, 1:] - d4[:, :1]
    return {"A3": np.arccos(np.clip(cosines, -1, 1)),
            "D1": np.linalg.norm(verts - verts.mean(axis=0), axis=1),
            "D2": np.linalg.norm(d2[:, 1] - d2[:, 0], axis=1),
            "D3": np.sqrt(np.linalg.norm(np.cross(d3[:, 1] - d3[:, 0], d3[:, 2] - d3[:, 0]), axis=1) / 2),
            "D4": np.cbrt(np.abs(np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))) / 6)}


def histogram(values, name):
    lo, hi = ft.ranges[name]
    counts, _ = np.histogram(np.clip(values, lo, hi), bins=ft.n_bins(name), range=(lo, hi))
    return counts / counts.sum()


def main():
    random.seed(0)
    paths = random.sample([p for p, _ in walk_shapes(normalised_dir)], shapes_used)
    hists = {(method, seed): {n: [] for n in names} for method in ("vertices", "surface") for seed in (0, 1)}
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(path)
        for seed in (0, 1):
            for method, values in (("vertices", vertex_values(mesh, np.random.default_rng(seed))),
                                   ("surface", ft.shape_property_values(mesh, np.random.default_rng(seed)))):
                for n in names:
                    hists[method, seed][n].append(histogram(values[n], n))
    l1 = lambda a, b: np.abs(a - b).sum()
    between = lambda h: np.mean([l1(h[i], h[j]) for i in range(len(h)) for j in range(i + 1, len(h))])
    rows = []
    for n in names:
        rows.append({"descriptor": n,
                     "seed noise, vertices": np.mean([l1(a, b) for a, b in zip(hists["vertices", 0][n], hists["vertices", 1][n])]),
                     "seed noise, surface": np.mean([l1(a, b) for a, b in zip(hists["surface", 0][n], hists["surface", 1][n])]),
                     "vertices vs surface": np.mean([l1(a, b) for a, b in zip(hists["vertices", 0][n], hists["surface", 0][n])]),
                     "signal, vertices": between(hists["vertices", 0][n]),
                     "signal, surface": between(hists["surface", 0][n])})
    table = pd.DataFrame(rows)
    table["vertices vs surface / signal"] = table["vertices vs surface"] / table["signal, surface"]
    save_stats(table, check_csv)
    print(f"{shapes_used} random shapes, {ft.samples:,} samples, {ft.bins} bins; all values are L1 distances between histograms")
    print(f"expected seed noise for independent samples: at most 2 sqrt(B / (pi N)) = {2 * np.sqrt(ft.bins / (np.pi * ft.samples)):.4f}")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
