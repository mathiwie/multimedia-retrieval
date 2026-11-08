import os
import random
import time
import numpy as np
import pandas as pd
import open3d as o3d

import features as ft
from preprocessing import out_dir, normalised_dir, walk_shapes, clean_mesh, save_stats

radii = [0.03, 0.05, 0.08, 0.12]
shapes_used = 10
choice_csv = os.path.join(out_dir, "curvature_choice.csv")


def one_ring_shape_index(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
    tris = tris[np.sort(first)]
    edges = [(tris[:, 1], tris[:, 2]), (tris[:, 2], tris[:, 0]), (tris[:, 0], tris[:, 1])]
    cots, angles = [], []
    for corner, (j, k) in enumerate(edges):
        i = tris[:, corner]
        u, v = verts[j] - verts[i], verts[k] - verts[i]
        cross = np.linalg.norm(np.cross(u, v), axis=1)
        dot = np.einsum("ij,ij->i", u, v)
        cots.append(np.divide(dot, cross, out=np.zeros(len(tris)), where=cross > 0))
        angles.append(np.arctan2(cross, dot))
    face_area = np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                                        verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1) / 2
    area, angle_sum, laplacian = np.zeros(len(verts)), np.zeros(len(verts)), np.zeros_like(verts)
    for corner in range(3):
        np.add.at(area, tris[:, corner], face_area / 3)
        np.add.at(angle_sum, tris[:, corner], angles[corner])
        j, k = edges[corner]
        weighted = cots[corner][:, None] * (verts[j] - verts[k])
        np.add.at(laplacian, j, -weighted)
        np.add.at(laplacian, k, weighted)
    ok = area > 0
    ends = np.concatenate([np.sort(np.stack([tris[:, a], tris[:, b]], 1), axis=1) for a, b in ((0, 1), (1, 2), (2, 0))])
    _, inverse, counts = np.unique(ends, axis=0, return_inverse=True, return_counts=True)
    ok[np.unique(ends[counts[inverse] < 2])] = False
    safe_area = np.where(ok, area, 1.0)
    single = o3d.geometry.TriangleMesh(mesh.vertices, o3d.utility.Vector3iVector(tris))
    single.orient_triangles()
    single.compute_vertex_normals()
    normals = np.asarray(single.vertex_normals)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        normals = -normals
    mean = -np.einsum("ij,ij->i", laplacian / (4 * safe_area[:, None]), normals)
    gauss = (2 * np.pi - angle_sum) / safe_area
    spread = np.sqrt(np.maximum(mean ** 2 - gauss, 0))
    k1, k2 = mean + spread, mean - spread
    flat = np.abs(k1) + np.abs(k2) < 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        shape_index = (2 / np.pi) * np.arctan((k1 + k2) / (k1 - k2))
    shape_index = np.where(flat | ~np.isfinite(shape_index), 0.0, shape_index)
    shape_index[~flat & (k1 == k2)] = np.sign(mean[~flat & (k1 == k2)])
    return shape_index[ok], np.ones(ok.sum())


def expected(k1, k2, radius):
    return np.sign(k1 + k2) * radius * np.sqrt(np.maximum(0.75 * (k1 ** 2 + k2 ** 2) - 0.5 * k1 * k2, 0))


def validate(radius):
    sphere_fine, sphere = ft.curvature_per_vertex(o3d.geometry.TriangleMesh.create_sphere(0.5, 100), radius)
    side = o3d.geometry.TriangleMesh.create_cylinder(0.3, 1.0, resolution=150, split=60)
    side.remove_triangles_by_mask(np.abs(np.asarray(side.vertices)[np.asarray(side.triangles)][:, :, 2]).max(axis=1) > 0.499)
    side.remove_unreferenced_vertices()
    side_fine, cylinder = ft.curvature_per_vertex(side, radius)
    middle = np.abs(np.asarray(side_fine.vertices)[:, 2]) < 0.35
    torus = o3d.geometry.TriangleMesh.create_torus(1.0, 0.3, 200, 80)
    torus.scale(0.5, center=(0, 0, 0))
    torus_fine, estimate = ft.curvature_per_vertex(torus, radius)
    v = np.asarray(torus_fine.vertices)
    cos_angle = (np.hypot(v[:, 0], v[:, 1]) / 0.5 - 1) / 0.3
    truth = expected(np.full(len(v), 1 / 0.15), cos_angle / (0.5 + 0.15 * cos_angle), radius)
    ok = np.isfinite(estimate)
    return {"radius": radius,
            "sphere": np.nanmedian(sphere), "sphere_expected": radius / 0.5,
            "cylinder": np.nanmedian(cylinder[middle]), "cylinder_expected": expected(1 / 0.3, 0, radius),
            "torus_error": np.mean(np.abs(estimate[ok] - truth[ok])),
            "torus_correlation": np.corrcoef(estimate[ok], truth[ok])[0, 1]}


def cube_band(radius, samples=4000):
    # every point closer than the radius to an edge of a cube sees part of the next face, so the
    # estimate spreads the curvature of the edges over a band; for a unit cube that band covers a
    # share 1 - (1 - 2 r)^2 of the surface
    cube = o3d.geometry.TriangleMesh.create_box().subdivide_midpoint(4)
    cube.translate(-cube.get_center())
    fine, verts, tris = ft.fine_mesh(cube, radius)
    area = ft.vertex_areas(verts, tris)
    query = np.unique(np.random.default_rng(ft.seed).choice(len(verts), size=samples, p=area / area.sum()))
    values = ft.curvature_at(fine, verts, tris, query, radius)
    values = values[np.isfinite(values)]
    return {"radius": radius, "band share": np.mean(np.abs(values) > 0.02), "expected": 1 - (1 - 2 * radius) ** 2}


def histogram(values, weights, lo, hi):
    counts, _ = np.histogram(np.clip(values, lo, hi), bins=ft.n_bins("CURV"), range=(lo, hi), weights=weights)
    return counts / weights.sum()


def robustness(meshes, estimator, lo, hi):
    hists, seconds = [], []
    for versions in meshes:
        start = time.perf_counter()
        row = [histogram(*estimator(m), lo, hi) for m in versions]
        seconds.append((time.perf_counter() - start) / len(versions))
        hists.append(row)
    subdivided = np.mean([np.abs(h[0] - h[1]).sum() for h in hists])
    decimated = np.mean([np.abs(h[0] - h[2]).sum() for h in hists])
    signal = np.mean([np.abs(hists[i][0] - hists[j][0]).sum()
                      for i in range(len(hists)) for j in range(i + 1, len(hists))])
    return {"subdivided": subdivided, "decimated": decimated, "signal": signal,
            "subdivided_ratio": subdivided / signal, "decimated_ratio": decimated / signal,
            "seconds_per_mesh": np.mean(seconds)}


def main():
    print("validation on shapes with known curvature (values are r * kappa)")
    print(f"{'r':>5s}  {'sphere':>15s}  {'cylinder side':>15s}  {'torus error':>11s}  {'torus corr':>10s}")
    for radius in sorted({0.05, 0.08, ft.curvature_radius}):
        v = validate(radius)
        print(f"{radius:>5.2f}  {v['sphere']:.4f} ({v['sphere_expected']:.4f})  "
              f"{v['cylinder']:.4f} ({v['cylinder_expected']:.4f})  {v['torus_error']:>11.4f}  {v['torus_correlation']:>10.4f}")

    print("\nshare of a cube's surface with a curvature above 0.02 (the band around the edges)")
    for radius in radii:
        b = cube_band(radius)
        print(f"  r = {radius:.2f}: {b['band share']:.3f} (expected {b['expected']:.3f})")

    random.seed(0)
    paths = random.sample([p for p, _ in walk_shapes(normalised_dir)], shapes_used)
    meshes = []
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(path)
        meshes.append((mesh, mesh.subdivide_midpoint(1), clean_mesh(mesh.simplify_quadric_decimation(len(mesh.triangles) // 2))))
    rows = [{"estimator": "one-ring shape index", **robustness(meshes, one_ring_shape_index, -1, 1)}]
    for radius in radii:
        estimator = lambda m, radius=radius: ft.curvature_values(m, radius=radius)[:2]
        rows.append({"estimator": f"covariance r = {radius}", **robustness(meshes, estimator, *ft.ranges["CURV"])})

    def finer(m, radius=ft.curvature_radius, edge=0.5):
        used, ft.curvature_edge = ft.curvature_edge, edge
        try:
            return ft.curvature_values(m, radius=radius)[:2]
        finally:
            ft.curvature_edge = used
    rows.append({"estimator": f"r = {ft.curvature_radius}, edges <= r/2", **robustness(meshes, finer, *ft.ranges["CURV"])})
    table = pd.DataFrame(rows)
    save_stats(table, choice_csv)
    print(f"\nsame shape at another resolution vs different shapes ({shapes_used} shapes, L1 between histograms)")
    print(f"{'estimator':24s} {'subdivided x4':>16s} {'decimated /2':>16s} {'signal':>7s} {'s/mesh':>7s}")
    for r in table.itertuples():
        print(f"{r.estimator:24s} {r.subdivided:6.3f} ({r.subdivided_ratio:6.1%}) {r.decimated:6.3f} ({r.decimated_ratio:6.1%})"
              f" {r.signal:7.3f} {r.seconds_per_mesh:7.2f}")
    worst = table.set_index("estimator")[["subdivided_ratio", "decimated_ratio"]].max(axis=1)
    print("\nworst-case noise / signal per estimator: " + ", ".join(f"{e} {w:.1%}" for e, w in worst.items()))
    print(f"used in features.py: r = {ft.curvature_radius} (worst case {worst[f'covariance r = {ft.curvature_radius}']:.1%})")


if __name__ == "__main__":
    main()
