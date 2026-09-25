import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d
from preprocessing import out_dir, normalised_dir, walk_shapes, save_stats, principal_axes

features_csv = os.path.join(out_dir, "features.csv")
skipped_csv = os.path.join(out_dir, "skipped_shapes.csv")

samples = 100_000   
bins = 40           
seed = 0            
volume_res = 128    

ranges = {
    "A3": (0.0, np.pi),
    "D1": (0.0, 1.0),
    "D2": (0.0, np.sqrt(3)),
    "D3": (0.0, np.sqrt(np.sqrt(3) / 2)),
    "D4": (0.0, (1 / 3) ** (1 / 3)),
    "SI": (-1.0, 1.0),
}

def curvature_values(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return np.array([]), 1.0
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

    area = np.zeros(len(verts))
    angle_sum = np.zeros(len(verts))
    laplacian = np.zeros_like(verts)
    for corner in range(3):
        np.add.at(area, tris[:, corner], face_area / 3)
        np.add.at(angle_sum, tris[:, corner], angles[corner])
        j, k = edges[corner]
        weighted = cots[corner][:, None] * (verts[j] - verts[k])
        np.add.at(laplacian, j, -weighted)
        np.add.at(laplacian, k, weighted)

    ok = area > 0
    ends = np.concatenate([np.sort(np.stack([tris[:, a], tris[:, b]], 1), axis=1)
                           for a, b in ((0, 1), (1, 2), (2, 0))])
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
    mean = -np.einsum("ij,ij->i", laplacian / (4 * safe_area[:, None]), normals) / 2
    gauss = (2 * np.pi - angle_sum) / safe_area

    spread = np.sqrt(np.maximum(mean ** 2 - gauss, 0))
    k1, k2 = mean + spread, mean - spread
    flat = np.abs(k1) + np.abs(k2) < 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        shape_index = (2 / np.pi) * np.arctan((k1 + k2) / (k1 - k2))
    shape_index = np.where(flat | ~np.isfinite(shape_index), 0.0, shape_index)
    shape_index[~flat & (k1 == k2)] = np.sign(mean[~flat & (k1 == k2)])
    return shape_index[ok], 1 - ok.mean()

def closed_mesh_volume(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)

def mesh_volume(mesh, res=volume_res):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    verts = np.asarray(mesh.vertices)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    step = (hi - lo).max() / res
    axes = [np.arange(lo[i] + step / 2, hi[i] + step / 2, step) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = scene.compute_occupancy(o3d.core.Tensor(points.astype(np.float32))).numpy()
    return float(inside.sum()) * step ** 3

def global_features(mesh):
    verts = np.asarray(mesh.vertices)
    area = mesh.get_surface_area()
    volume = mesh_volume(mesh)
    hull, _ = mesh.compute_convex_hull()
    hull_verts = np.asarray(hull.vertices)
    hull_volume = closed_mesh_volume(hull)
    diameter = np.sqrt(((hull_verts[:, None, :] - hull_verts[None, :, :]) ** 2).sum(-1).max())

    leaked = volume > hull_volume
    volume = min(volume, hull_volume)
    compactness = area ** 3 / (36 * np.pi * volume ** 2) if volume > 0 else np.nan

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
        "volume_doubtful": bool(leaked or compactness < 1),
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

    d1 = np.linalg.norm(pick(1)[:, 0] - verts.mean(axis=0), axis=1)

    d2_pts = pick(2)
    d2 = np.linalg.norm(d2_pts[:, 1] - d2_pts[:, 0], axis=1)

    d3_pts = pick(3)
    cross = np.cross(d3_pts[:, 1] - d3_pts[:, 0], d3_pts[:, 2] - d3_pts[:, 0])
    d3 = np.sqrt(np.linalg.norm(cross, axis=1) / 2)

    d4_pts = pick(4)
    edges = d4_pts[:, 1:] - d4_pts[:, :1]
    d4 = np.cbrt(np.abs(np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))) / 6)

    return {"A3": angles, "D1": d1, "D2": d2, "D3": d3, "D4": d4}

def histogram_features(values):
    row = {}
    for name, data in values.items():
        lo, hi = ranges[name]
        counts, _ = np.histogram(np.clip(data, lo, hi), bins=bins, range=ranges[name])
        for i, count in enumerate(counts):
            row[f"{name}_{i}"] = count / len(data)
    return row

def extract_features(path, shape_class=""):
    mesh = o3d.io.read_triangle_mesh(path)
    if np.asarray(mesh.vertices).shape[0] == 0:
        raise ValueError("empty mesh")
    rng = np.random.default_rng(seed)
    row = {"path": path, "class": shape_class}
    row.update(global_features(mesh))
    values = shape_property_values(mesh, rng)
    shape_index, dropped = curvature_values(mesh)
    if len(shape_index) == 0:
        raise ValueError("no vertex with a usable curvature")
    values["SI"] = shape_index
    row.update(histogram_features(values))
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