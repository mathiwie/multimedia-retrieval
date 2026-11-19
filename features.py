import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from preprocessing import (out_dir, normalised_dir, walk_shapes, save_stats, principal_axes, split_long_edges,
                           area_barycenter)

features_csv = os.path.join(out_dir, "features.csv")
skipped_csv = os.path.join(out_dir, "skipped_shapes.csv")

samples = 100_000
bins = 40
# odd,so that flat regions (value 0) get a bin of their own in the middle
curvature_bins = 41
seed = 0
volume_res = 128
volume_rays = 5
leak_limit = 1.0
curvature_max = np.sqrt(8)
curvature_radius = 0.05
curvature_min_faces = 4
curvature_flat = 1e-3
# split edges longer than the patch radius before the curvature
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


def n_bins(name):
    return curvature_bins if name == "CURV" else bins


def bin_centres(name):
    lo, hi = ranges[name]
    return lo + (np.arange(n_bins(name)) + 0.5) * (hi - lo) / n_bins(name)


def surface_graph(verts, tris):
    ends = np.unique(np.sort(tris[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    lengths = np.maximum(np.linalg.norm(verts[ends[:, 0]] - verts[ends[:, 1]], axis=1), 1e-12)
    n = len(verts)
    return csr_matrix((np.r_[lengths, lengths], (np.r_[ends[:, 0], ends[:, 1]], np.r_[ends[:, 1], ends[:, 0]])),
                      shape=(n, n))


def orient_outward(mesh):
    # one copy of every face, all triangles of a connected part oriented consistently and every part
    # turned so that its normals point outwards
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
    single = o3d.geometry.TriangleMesh(mesh.vertices, o3d.utility.Vector3iVector(tris[np.sort(first)]))
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
    return single


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
    # orient first and split afterwards since splitting keeps the winding of
    # every triangle and orienting the coarse mesh is much faster than
    # orienting the fine one
    fine = split_long_edges(orient_outward(mesh), radius * curvature_edge)
    fine.compute_vertex_normals()
    return fine, np.asarray(fine.vertices), np.asarray(fine.triangles)


def reachable(graph, verts, query, rows, targets, limit, cell, chunk=500):
    # for every pair (query[rows[i]], targets[i]) can the target vertex be
    # reached from the query vertex along mesh edges with a path of at most limit?
    _, group = np.unique(np.floor(verts[query] / cell).astype(np.int64), axis=0, return_inverse=True)
    group = group.ravel()
    by_cell = np.argsort(group, kind="stable")
    starts = np.flatnonzero(np.r_[True, group[by_cell][1:] != group[by_cell][:-1]])
    batches = [by_cell[i:min(i + chunk, end)]
               for start, end in zip(starts, np.r_[starts[1:], len(query)])
               for i in range(start, end, chunk)]
    batch_of = np.empty(len(query), dtype=np.int64)
    for i, part in enumerate(batches):
        batch_of[part] = i
    pair_batch = batch_of[rows]
    order = np.argsort(pair_batch, kind="stable")
    bounds = np.searchsorted(pair_batch[order], np.arange(len(batches) + 1))

    found = np.zeros(len(rows), dtype=bool)
    row_of = np.full(len(query), -1)
    local = np.full(len(verts), -1)
    for i, part in enumerate(batches):
        pairs = order[bounds[i]:bounds[i + 1]]
        if len(pairs) == 0:
            continue
        points = verts[query[part]]
        lo, hi = points.min(axis=0) - limit, points.max(axis=0) + limit
        inside = np.flatnonzero(np.all((verts >= lo) & (verts <= hi), axis=1))
        local[:] = -1
        local[inside] = np.arange(len(inside))
        dist = dijkstra(graph[inside][:, inside], indices=local[query[part]], limit=limit)
        row_of[part] = np.arange(len(part))
        target = local[targets[pairs]]
        ok = target >= 0
        found[pairs[ok]] = np.isfinite(dist[row_of[rows[pairs[ok]]], target[ok]])
    return found


def curvature_at(mesh, verts, tris, query, radius, block=1000):
    graph = surface_graph(verts, tris)
    normals = np.asarray(mesh.vertex_normals)
    face_area, first_moment, second_moment = face_moments(verts, tris)
    faces = cKDTree(verts[tris].mean(axis=1))
    values = np.full(len(query), np.nan)
    spatial = np.lexsort(np.floor(verts[query] / (2 * radius)).T[::-1])
    for start in range(0, len(query), block):
        part = spatial[start:start + block]
        points = query[part]
        close = cKDTree(verts[points]).sparse_distance_matrix(faces, radius, output_type="coo_matrix")
        keep = reachable(graph, verts, points, close.row, tris[close.col, 0], 2 * radius, 2 * radius)
        near = csr_matrix((np.ones(keep.sum()), (close.row[keep], close.col[keep])), shape=(len(points), len(tris)))
        del close, keep

        total = near @ face_area
        usable = (np.diff(near.indptr) >= curvature_min_faces) & (total > 0)
        total = np.where(usable, total, 1.0)
        mean = near @ first_moment / total[:, None]
        cov = (near @ second_moment / total[:, None]).reshape(-1, 3, 3) - mean[:, :, None] * mean[:, None, :]
        eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
        spread = eig.sum(axis=1)
        variation = np.divide(eig[:, 0], spread, out=np.zeros(len(points)), where=spread > 0)
        side = -np.sign(np.einsum("ij,ij->i", mean - verts[points], normals[points]))
        magnitude = np.sqrt(24 * variation)
        part_values = np.where(magnitude < curvature_flat, 0.0, side * magnitude)
        values[part] = np.where(usable, part_values, np.nan)
    return values


def curvature_values(mesh, radius=curvature_radius, samples=curvature_samples):
    mesh, verts, tris = fine_mesh(mesh, radius)
    if len(tris) == 0:
        return np.array([]), np.array([]), 1.0
    # vertices drawn with a probability proportional to their area, so every
    # histogram rests on the same number of area uniform samples
    area = vertex_areas(verts, tris)
    drawn = np.random.default_rng(seed).choice(len(verts), size=samples,
                                               p=area / area.sum())
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


def golden_spiral(n):
    # n directions spread evenly over the sphere - the heights z go down in equal steps which gives every
    # direction a band of the sphere with the same area and every next direction is turned by the
    # golden angle (137.5 degrees) around the z axis so that no two of them line up
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = i * np.pi * (3 - np.sqrt(5))
    return np.stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z], axis=1)


def ray_directions(rays=volume_rays):
    # golden spiral!!! -> the directions are spread as evenly as possible over the sphere, so one hole
    # is unlikely to fool several rays at once also the fixed random rotation makes sure that no ray is
    # parallel to a coordinate axis along which the faces of aligned shapes and the voxel grid lie
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return golden_spiral(rays) @ rotation.T


def voxel_grid(mesh, res=volume_res):
    verts = np.asarray(mesh.vertices)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    step = (hi - lo).max() / res
    axes = [np.arange(lo[i] + step / 2, hi[i] + step / 2, step) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    return axes, points, step


def voxel_votes(mesh, res=volume_res, rays=volume_rays):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    _, points, step = voxel_grid(mesh, res)
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


def surface_points(verts, tris, n, rng):
    # n points spread uniformly over the surface: a face is picked with a probability proportional
    # to its area, then a uniformly random point inside it
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    face = rng.choice(len(tris), size=n, p=area / area.sum())
    s, t = np.sqrt(rng.random(n)), rng.random(n)
    return (1 - s)[:, None] * a[face] + (s * (1 - t))[:, None] * b[face] + (s * t)[:, None] * c[face]


def shape_property_values(mesh, rng):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    pick = lambda k: surface_points(verts, tris, samples * k, rng).reshape(samples, k, 3)

    a3 = pick(3)
    u = a3[:, 1] - a3[:, 0]
    v = a3[:, 2] - a3[:, 0]
    lengths = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cosines = np.divide(np.einsum("ij,ij->i", u, v), lengths,
                        out=np.zeros(samples), where=lengths > 0)
    angles = np.arccos(np.clip(cosines, -1, 1))

    d1 = np.linalg.norm(pick(1)[:, 0] - area_barycenter(mesh), axis=1)

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
        counts, _ = np.histogram(np.clip(data, lo, hi), bins=n_bins(name), range=ranges[name], weights=w)
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
