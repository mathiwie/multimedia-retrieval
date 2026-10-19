import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from preprocessing import out_dir, normalised_dir, walk_shapes, save_stats, principal_axes, split_long_edges

features_csv = os.path.join(out_dir, "features.csv")
skipped_csv = os.path.join(out_dir, "skipped_shapes.csv")

samples = 100_000   
bins = 40           
seed = 0            
volume_res = 128
volume_rays = 5
leak_limit = 1.0
curvature_max = np.sqrt(8)
curvature_radius = 0.08
curvature_min_faces = 4
curvature_flat = 1e-3
curvature_edge = 1.0
curvature_samples = 10_000

ranges = {
    "A3": (0.0, np.pi),
    "D1": (0.0, 1.0),
    "D2": (0.0, np.sqrt(3)),
    "D3": (0.0, np.sqrt(np.sqrt(3) / 2)),
    "D4": (0.0, (1 / 3) ** (1 / 3)),
    "CURV": (-curvature_max, curvature_max),
}

def surface_graph(verts, tris):
    ends = np.unique(np.sort(tris[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    lengths = np.maximum(np.linalg.norm(verts[ends[:, 0]] - verts[ends[:, 1]], axis=1), 1e-12)
    n = len(verts)
    return csr_matrix((np.r_[lengths, lengths], (np.r_[ends[:, 0], ends[:, 1]], np.r_[ends[:, 1], ends[:, 0]])),
                      shape=(n, n))

def outward_normals(mesh, tris):
    verts = np.asarray(mesh.vertices)
    single = o3d.geometry.TriangleMesh(mesh.vertices, o3d.utility.Vector3iVector(tris))
    single.orient_triangles()
    labels = np.asarray(single.cluster_connected_triangles()[0])
    oriented = np.asarray(single.triangles).copy()
    corners = verts[oriented]
    centroids = corners.mean(axis=1)
    count = np.bincount(labels)
    centre = np.stack([np.bincount(labels, centroids[:, i]) / count for i in range(3)], axis=1)[labels]
    a, b, c = (corners[:, i] - centre for i in range(3))
    volume = np.bincount(labels, np.einsum("ij,ij->i", a, np.cross(b, c)))
    inside_out = volume[labels] < 0
    oriented[inside_out] = oriented[inside_out][:, [0, 2, 1]]
    single.triangles = o3d.utility.Vector3iVector(oriented)
    single.compute_vertex_normals()
    return np.asarray(single.vertex_normals)

def vertex_areas(verts, tris):
    face_area = np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                                        verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1) / 2
    area = np.zeros(len(verts))
    for corner in range(3):
        np.add.at(area, tris[:, corner], face_area / 3)
    return area

def face_moments(verts, tris):
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    total = a + b + c
    second = sum(v[:, :, None] * v[:, None, :] for v in (a, b, c, total))
    return area, area[:, None] * total / 3, (area[:, None, None] * second / 12).reshape(-1, 9)

def fine_mesh(mesh, radius):
    mesh = split_long_edges(mesh, radius * curvature_edge)
    tris = np.asarray(mesh.triangles)
    _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
    return mesh, np.asarray(mesh.vertices), tris[np.sort(first)]

def curvature_at(mesh, verts, tris, query, radius, chunk=500):
    graph = surface_graph(verts, tris)
    normals = outward_normals(mesh, tris)[query]
    face_area, first_moment, second_moment = face_moments(verts, tris)
    centroids = verts[tris].mean(axis=1)

    close = cKDTree(verts[query]).sparse_distance_matrix(cKDTree(centroids), radius, output_type="coo_matrix")
    keep = np.zeros(close.nnz, dtype=bool)
    order = np.argsort(close.row, kind="stable")
    rows_sorted, faces_sorted = close.row[order], close.col[order]
    bounds = np.searchsorted(rows_sorted, np.arange(0, len(query) + chunk, chunk))
    for c, start in enumerate(range(0, len(query), chunk)):
        lo, hi = bounds[c], bounds[c + 1]
        if lo == hi:
            continue
        reach = np.isfinite(dijkstra(graph, indices=query[start:start + chunk], limit=2 * radius))
        keep[order[lo:hi]] = reach[rows_sorted[lo:hi] - start, tris[faces_sorted[lo:hi], 0]]
    near = csr_matrix((np.ones(keep.sum()), (close.row[keep], close.col[keep])), shape=(len(query), len(tris)))

    total = near @ face_area
    usable = (np.diff(near.indptr) >= curvature_min_faces) & (total > 0)
    total = np.where(usable, total, 1.0)
    mean = near @ first_moment / total[:, None]
    cov = (near @ second_moment / total[:, None]).reshape(-1, 3, 3) - mean[:, :, None] * mean[:, None, :]
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    spread = eig.sum(axis=1)
    variation = np.divide(eig[:, 0], spread, out=np.zeros(len(query)), where=spread > 0)
    side = -np.sign(np.einsum("ij,ij->i", mean - verts[query], normals))
    magnitude = np.sqrt(24 * variation)
    values = np.where(magnitude < curvature_flat, 0.0, side * magnitude)
    return np.where(usable, values, np.nan)

def curvature_values(mesh, radius=curvature_radius, samples=curvature_samples):
    mesh, verts, tris = fine_mesh(mesh, radius)
    if len(tris) == 0:
        return np.array([]), np.array([]), 1.0
    area = vertex_areas(verts, tris)
    if len(verts) <= samples:
        query, weights = np.flatnonzero(area > 0), area[area > 0]
    else:
        drawn = np.random.default_rng(seed).choice(len(verts), size=samples, p=area / area.sum())
        query, counts = np.unique(drawn, return_counts=True)
        weights = counts.astype(float)
    values = curvature_at(mesh, verts, tris, query, radius)
    usable = np.isfinite(values)
    return values[usable], weights[usable], 1 - weights[usable].sum() / weights.sum()

def curvature_per_vertex(mesh, radius=curvature_radius):
    mesh, verts, tris = fine_mesh(mesh, radius)
    return mesh, curvature_at(mesh, verts, tris, np.arange(len(verts)), radius)

def closed_mesh_volume(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)

def ray_directions(rays=volume_rays):
    d = np.random.default_rng(seed).normal(size=(rays, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)

def voxel_votes(mesh, res=volume_res, rays=volume_rays):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    verts = np.asarray(mesh.vertices)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    step = (hi - lo).max() / res
    axes = [np.arange(lo[i] + step / 2, hi[i] + step / 2, step) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    votes = np.zeros((rays, len(points)), dtype=bool)
    for i, d in enumerate(ray_directions(rays)):
        cast = np.hstack([points, np.broadcast_to(d.astype(np.float32), points.shape)])
        votes[i] = scene.count_intersections(o3d.core.Tensor(cast)).numpy() % 2 == 1
    return votes, step ** 3

def leakage(votes):
    count = votes.sum(axis=0)
    possible = count > 0
    return float((possible & (count < len(votes))).sum() / possible.sum()) if possible.any() else 1.0

def mesh_volume(mesh, res=volume_res, rays=volume_rays):
    votes, cell = voxel_votes(mesh, res, rays)
    return float((votes.sum(axis=0) > rays // 2).sum()) * cell, leakage(votes)

def settle_volume(volume, leak, area, hull_volume, hull_area, limit=None):
    limit = leak_limit if limit is None else limit
    compactness = area ** 3 / (36 * np.pi * volume ** 2) if volume > 0 else np.nan
    fallback = not volume > 0 or volume > hull_volume or leak > limit or compactness < 1
    if fallback:
        volume = hull_volume
        compactness = hull_area ** 3 / (36 * np.pi * volume ** 2) if volume > 0 else np.nan
    return volume, compactness, fallback

def global_features(mesh):
    area = mesh.get_surface_area()
    voxel_volume, leak = mesh_volume(mesh)
    hull, _ = mesh.compute_convex_hull()
    hull_verts = np.asarray(hull.vertices)
    hull_volume = closed_mesh_volume(hull)
    diameter = np.sqrt(((hull_verts[:, None, :] - hull_verts[None, :, :]) ** 2).sum(-1).max())
    volume, compactness, hull_fallback = settle_volume(voxel_volume, leak, area, hull_volume, hull.get_surface_area())

    obb_volume = mesh.get_oriented_bounding_box().volume()
    eigenvalues, _ = principal_axes(mesh)

    return {
        "area": area,
        "compactness": compactness,
        "rectangularity": volume / obb_volume if obb_volume > 0 else np.nan,
        "diameter": diameter,
        "convexity": volume / hull_volume if hull_volume > 0 else np.nan,
        "eccentricity": eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 0 else np.nan,
        "volume": volume,
        "leakage": leak,
        "hull_fallback": bool(hull_fallback),
    }

def shape_property_values(mesh, rng):
    verts = np.asarray(mesh.vertices)
    n = verts.shape[0]
    pick = lambda k: verts[rng.integers(0, n, size=(samples, k))]

    a3 = pick(3)
    u = a3[:, 1] - a3[:, 0]
    v = a3[:, 2] - a3[:, 0]
    lengths = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cosines = np.divide(np.einsum("ij,ij->i", u, v), lengths,
                        out=np.zeros(samples), where=lengths > 0)
    angles = np.arccos(np.clip(cosines, -1, 1))

    d1 = np.linalg.norm(verts - verts.mean(axis=0), axis=1)

    d2_pts = pick(2)
    d2 = np.linalg.norm(d2_pts[:, 1] - d2_pts[:, 0], axis=1)

    d3_pts = pick(3)
    cross = np.cross(d3_pts[:, 1] - d3_pts[:, 0], d3_pts[:, 2] - d3_pts[:, 0])
    d3 = np.sqrt(np.linalg.norm(cross, axis=1) / 2)

    d4_pts = pick(4)
    edges = d4_pts[:, 1:] - d4_pts[:, :1]
    d4 = np.cbrt(np.abs(np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))) / 6)

    return {"A3": angles, "D1": d1, "D2": d2, "D3": d3, "D4": d4}

def histogram_features(values, weights={}):
    row = {}
    for name, data in values.items():
        lo, hi = ranges[name]
        w = weights.get(name, np.ones(len(data)))
        counts, _ = np.histogram(np.clip(data, lo, hi), bins=bins, range=ranges[name], weights=w)
        for i, count in enumerate(counts):
            row[f"{name}_{i}"] = count / w.sum()
    return row

def extract_features(path, shape_class=""):
    mesh = o3d.io.read_triangle_mesh(path)
    if np.asarray(mesh.vertices).shape[0] == 0:
        raise ValueError("empty mesh")
    rng = np.random.default_rng(seed)
    row = {"path": path, "class": shape_class}
    row.update(global_features(mesh))
    values = shape_property_values(mesh, rng)
    curvature, area, dropped = curvature_values(mesh)
    if len(curvature) == 0 or area.sum() == 0:
        raise ValueError("no vertex with a usable curvature")
    values["CURV"] = curvature
    row.update(histogram_features(values, {"CURV": area}))
    row["curvature_dropped"] = dropped
    row["curvature_doubtful"] = bool(dropped > 0.5)
    return row

def extract_all(root=normalised_dir):
    rows, skipped = [], []
    for i, (path, cls) in enumerate(walk_shapes(root), start=1):
        try:
            rows.append(extract_features(path, cls))
        except Exception as e:
            print("couldnt extract:", path, "-", e)
            skipped.append({"path": path, "class": cls, "reason": str(e)})
        if i % 250 == 0:
            print(f"  {i} shapes done")
    return pd.DataFrame(rows), pd.DataFrame(skipped)

def print_features(key):
    path = os.path.join(normalised_dir, key + ".obj")
    if os.path.isfile(features_csv):
        saved = pd.read_csv(features_csv)
        match = saved[saved["path"].str.replace("\\", "/", regex=False) == path.replace("\\", "/")]
        if len(match) > 0:
            print(f"{key} (from {features_csv})")
            print(match.iloc[0].to_string())
            return
    if not os.path.isfile(path):
        sys.exit(f"shape not found: {path}")
    print(f"{key} (computed now)")
    print(pd.Series(extract_features(path, os.path.dirname(key))).to_string())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_features(sys.argv[1])
    else:
        print("extracting features from", normalised_dir)
        features, skipped = extract_all()
        save_stats(features, features_csv)
        save_stats(skipped, skipped_csv)
        print(features[["area", "compactness", "rectangularity", "diameter",
                        "convexity", "eccentricity"]].describe())