import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

db_path = "ShapeDatabase_INFOMR"
out_dir = "out"
resampled_dir = os.path.join(out_dir, "resampled")
stats_csv = os.path.join(out_dir, "shape_stats.csv")
stats_csv_after = os.path.join(out_dir, "shape_stats_after_resample.csv")
normalised_dir = os.path.join(out_dir, "normalised")
average_outliers_csv = os.path.join(out_dir, "average_and_outliers.csv")
normalisation_csv = os.path.join(out_dir, "normalisation_check.csv")
normalisation_before_csv = os.path.join(out_dir, "normalisation_check_before.csv")
small_thresh = 100
big_thresh = 50000
target_verts = 7500
target_min = 5000
target_max = 10000


def read_obj(path):
    verts, tris, sizes = [], [], Counter()
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append(parts[1:4])
            elif parts[0] == "f":
                index = [int(p.split("/")[0]) for p in parts[1:]]
                index = [i - 1 if i > 0 else len(verts) + i for i in index]
                sizes[len(index)] += 1
                tris.extend([index[0], index[k], index[k + 1]] for k in range(1, len(index) - 1))
    return (np.array(verts, dtype=float).reshape(-1, 3), np.array(tris, dtype=np.int64).reshape(-1, 3), sizes)


def face_type_of(sizes):
    if not sizes:
        return "none"
    if set(sizes) == {3}:
        return "triangles"
    if set(sizes) == {4}:
        return "quads"
    return "mixed"


def file_counts(path):
    # number of vertices and faces stored in the file
    verts, _, sizes = read_obj(path)
    return len(verts), sum(sizes.values())


def analyze_shape(path, shape_class):
    verts, _, sizes = read_obj(path)
    mesh = o3d.io.read_triangle_mesh(path)
    loaded = np.asarray(mesh.vertices)
    # the bounding box is that of the surface, so vertices that no face uses do not count
    if loaded.shape[0] > 0:
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_min = bbox.get_min_bound()
        bbox_max = bbox.get_max_bound()
    else:
        bbox_min = np.array([np.nan] * 3)
        bbox_max = np.array([np.nan] * 3)
    return {
        "path": path.replace("\\", "/"),
        "class": shape_class,
        "num_verts": len(verts),
        "num_faces": sum(sizes.values()),
        "face_type": face_type_of(sizes),
        "loaded_verts": loaded.shape[0],
        "loaded_faces": np.asarray(mesh.triangles).shape[0],
        "min_x": bbox_min[0], "min_y": bbox_min[1], "min_z": bbox_min[2],
        "max_x": bbox_max[0], "max_y": bbox_max[1], "max_z": bbox_max[2],
    }


def walk_shapes(root):
    for class_name in sorted(os.listdir(root)):
        class_folder = os.path.join(root, class_name)
        if not os.path.isdir(class_folder):
            continue
        objs = sorted(glob.glob(os.path.join(class_folder, "**", "*.obj"), recursive=True))
        for obj_path in objs:
            yield obj_path, class_name


def make_stats_df(root):
    data = []
    for path, cls in walk_shapes(root):
        try:
            data.append(analyze_shape(path, cls))
        except Exception as e:
            print("couldnt read:", path, "-", e)
    return pd.DataFrame(data)


def save_stats(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {len(df)} rows to {path}")


def make_histograms(df, label=""):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(df["num_verts"], bins=50, color="steelblue", edgecolor="black")
    ax[0].set_title("vertex counts " + label)
    ax[0].set_xlabel("# verts")
    ax[0].set_ylabel("# shapes")
    ax[1].hist(df["num_faces"], bins=50, color="indianred", edgecolor="black")
    ax[1].set_title("face counts " + label)
    ax[1].set_xlabel("# faces")
    ax[1].set_ylabel("# shapes")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"histograms_{label.strip().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved histogram to", fname)


def make_class_histogram(df, label=""):
    os.makedirs(out_dir, exist_ok=True)
    counts = df["class"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, max(5, len(counts) * 0.16)))
    ax.barh(counts.index, counts.values, color="seagreen", edgecolor="black")
    ax.set_title("shapes per class")
    ax.set_xlabel("# shapes")
    ax.set_ylabel("class")
    ax.tick_params(axis="y", labelsize=7)
    ax.margins(y=0.005)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"class_histogram_{label.strip().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved class histogram to", fname)


def print_stats_summary(df):
    print("\nsummary (vertices and faces as stored in the files)")
    print("num shapes:", len(df))
    print("avg verts:", round(df["num_verts"].mean(), 1))
    print("avg faces:", round(df["num_faces"].mean(), 1))
    print("smallest:", df.loc[df["num_verts"].idxmin(), "path"],
          "-", df["num_verts"].min(), "verts")
    print("biggest:", df.loc[df["num_verts"].idxmax(), "path"],
          "-", df["num_verts"].max(), "verts")
    merged = df["num_verts"] - df["loaded_verts"]
    print(f"open3d merges or drops {merged.sum():,} vertices on loading, in {(merged > 0).sum()} shapes")


def find_average_and_outliers(df, n=3, lo=small_thresh, hi=big_thresh):
    dv = (df["num_verts"] - df["num_verts"].mean()) / df["num_verts"].std()
    dfc = (df["num_faces"] - df["num_faces"].mean()) / df["num_faces"].std()
    average = df.loc[[(dv ** 2 + dfc ** 2).idxmin()]]
    small = df[(df["num_verts"] < lo) | (df["num_faces"] < lo)]
    big = df[(df["num_verts"] > hi) | (df["num_faces"] > hi)]
    picked = pd.concat([
        average.assign(kind="average"),
        small.nsmallest(n, "num_verts").assign(kind="small outlier"),
        big.nlargest(n, "num_verts").assign(kind="big outlier"),
    ])
    print(f"average shape: {average['path'].iloc[0]} "
          f"({average['num_verts'].iloc[0]}v / {average['num_faces'].iloc[0]}f)")
    print(f"small outliers (<{lo} verts or faces): {len(small)}")
    print(f"big outliers (>{hi} verts or faces): {len(big)}")
    return picked


def remove_reversed_triangles(mesh):
    # open3d's remove_duplicated_triangles only removes copies with the same winding,
    # so a triangle stored once per side, (i, j, k) and (i, k, j), survives it
    tris = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return mesh
    _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
    if len(first) < len(tris):
        drop = np.ones(len(tris), dtype=bool)
        drop[first] = False
        mesh.remove_triangles_by_mask(drop)
    return mesh


def clean_mesh(mesh):
    if num_verts(mesh) > 0:
        extent = np.max(mesh.get_max_bound() - mesh.get_min_bound())
        mesh.merge_close_vertices(extent * 1e-6)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    remove_reversed_triangles(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def num_verts(mesh):
    return np.asarray(mesh.vertices).shape[0]


def search_in_range(simplify, p_min, p_max, target=target_verts, lo=target_min, hi=target_max, iters=12):
    best = None
    for _ in range(iters):
        p = (p_min + p_max) / 2
        result = clean_mesh(simplify(p))
        n = num_verts(result)
        if best is None or abs(n - target) < abs(num_verts(best) - target):
            best = result
        if lo <= n <= hi:
            return result
        if n > hi:
            p_max = p
        else:
            p_min = p
    return best

def simplify_mesh(mesh, target=target_verts, lo=target_min, hi=target_max):
    decimate = lambda tris: mesh.simplify_quadric_decimation(target_number_of_triangles=max(int(tris), 4))
    simplified = clean_mesh(decimate(target * 2))
    if lo <= num_verts(simplified) <= hi:
        return simplified
    simplified = search_in_range(decimate, 4, np.asarray(mesh.triangles).shape[0], target, lo, hi)
    if num_verts(simplified) <= hi:
        return simplified
    print(f"  quadric decimation stuck at {num_verts(simplified)}v, using vertex clustering")
    extent = np.max(mesh.get_max_bound() - mesh.get_min_bound())
    cluster = lambda divisions: mesh.simplify_vertex_clustering(voxel_size=extent / divisions)
    return search_in_range(cluster, 10, 1000, target, lo, hi)


def mesh_edges(tris):
    ends = np.sort(tris[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    keys, edge_id = np.unique(ends, axis=0, return_inverse=True)
    return keys, edge_id.reshape(-1, 3)


def split_edges(verts, tris, keys, edge_id, marked):
    while True:
        count = marked[edge_id].sum(axis=1)
        promote = (count == 2)
        if not promote.any():
            break
        marked[edge_id[promote].ravel()] = True

    midpoint = np.full(len(keys), -1)
    midpoint[marked] = len(verts) + np.arange(marked.sum())
    new_verts = np.vstack([verts, verts[keys[marked]].mean(axis=1)])
    m = midpoint[edge_id]

    keep = tris[count == 0]
    red_t, red_m = tris[count == 3], m[count == 3]
    red = np.concatenate([
        np.stack([red_t[:, 0], red_m[:, 0], red_m[:, 2]], 1),
        np.stack([red_m[:, 0], red_t[:, 1], red_m[:, 1]], 1),
        np.stack([red_m[:, 2], red_m[:, 1], red_t[:, 2]], 1),
        np.stack([red_m[:, 0], red_m[:, 1], red_m[:, 2]], 1)])
    green_t, green_m = tris[count == 1], m[count == 1]
    j = np.argmax(green_m >= 0, axis=1)
    rows = np.arange(len(green_t))
    va, vb, vc = green_t[rows, j], green_t[rows, (j + 1) % 3], green_t[rows, (j + 2) % 3]
    mid = green_m[rows, j]
    green = np.concatenate([np.stack([va, mid, vc], 1), np.stack([mid, vb, vc], 1)])

    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(new_verts),
                                     o3d.utility.Vector3iVector(np.concatenate([keep, red, green])))

def split_largest(mesh, budget):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    keys, edge_id = mesh_edges(tris)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    order = np.argsort(-np.linalg.norm(np.cross(b - a, c - a), axis=1), kind="stable")

    _, first = np.unique(edge_id[order].ravel(), return_index=True)
    new_edges = np.cumsum(np.bincount(first // 3, minlength=len(order)))
    chosen = order[:max(1, np.searchsorted(new_edges, budget, side="right"))]
    marked = np.zeros(len(keys), dtype=bool)
    marked[edge_id[chosen].ravel()] = True
    return split_edges(verts, tris, keys, edge_id, marked)


def split_long_edges(mesh, max_length, rounds=20):
    for _ in range(rounds):
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        keys, edge_id = mesh_edges(tris)
        marked = np.linalg.norm(verts[keys[:, 0]] - verts[keys[:, 1]], axis=1) > max_length
        if not marked.any():
            break
        mesh = split_edges(verts, tris, keys, edge_id, marked)
    return mesh


def refine_mesh(mesh, target=target_verts, lo=target_min, hi=target_max, rounds=30):
    count = 0
    while num_verts(mesh) < lo and count < rounds:
        mesh = clean_mesh(split_largest(mesh, target - num_verts(mesh)))
        count += 1
    if num_verts(mesh) > hi:
        mesh = simplify_mesh(mesh, target, lo, hi)
    return mesh


def fix_outliers(df, root, dest, lo=target_min, hi=target_max):
    os.makedirs(dest, exist_ok=True)
    for _, row in df.iterrows():
        path = row["path"]
        cls = row["class"]
        try:
            mesh = clean_mesh(o3d.io.read_triangle_mesh(path))
            if num_verts(mesh) == 0:
                print(f"skipping {path} (empty mesh, 0 verts)")
                continue
            nv = num_verts(mesh)
            nf = np.asarray(mesh.triangles).shape[0]
            if nv < lo:
                mesh = refine_mesh(mesh)
            elif nv > hi:
                mesh = simplify_mesh(mesh)
            new_nv = num_verts(mesh)
            if not lo <= new_nv <= hi:
                print(f"out of range after resampling: {path} ({nv}v / {nf}f -> {new_nv}v)")
            cls_dir = os.path.join(dest, cls)
            os.makedirs(cls_dir, exist_ok=True)
            out_path = os.path.join(cls_dir, os.path.basename(path))
            o3d.io.write_triangle_mesh(out_path, mesh)
        except Exception as e:
            print("couldnt resample:", path, "-", e)
    print("done resampling:", dest)


def area_weighted_covariance(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    total = areas.sum()
    if total == 0:
        return np.cov(verts.T)
    weights = areas / total
    centroids = (a + b + c) / 3
    offsets = centroids - (weights[:, None] * centroids).sum(axis=0)
    return np.einsum("i,ij,ik->jk", weights, offsets, offsets)


def principal_axes(mesh):
    eigenvalues, eigenvectors = np.linalg.eigh(area_weighted_covariance(mesh))
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def area_barycenter(mesh):
    # so it does not depend on how the vertices are spread over the surface (feedback fix)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return verts.mean(axis=0)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    if areas.sum() == 0:
        return verts.mean(axis=0)
    return (areas[:, None] * (a + b + c) / 3).sum(axis=0) / areas.sum()


def flip_moments(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    centroids = verts[tris].mean(axis=1)
    return np.sum(np.sign(centroids) * centroids ** 2, axis=0)


def normalisation_steps(mesh):
    steps = [("original", o3d.geometry.TriangleMesh(mesh))]
    verts = np.asarray(mesh.vertices)
    verts = verts - area_barycenter(mesh)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    steps.append(("translated", o3d.geometry.TriangleMesh(mesh)))

    _, axes = principal_axes(mesh)
    rotation = np.stack([axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])])
    verts = verts @ rotation.T
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    steps.append(("aligned", o3d.geometry.TriangleMesh(mesh)))

    signs = np.where(flip_moments(mesh) < 0, -1.0, 1.0)
    verts = verts * signs
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    if np.prod(signs) < 0:
        tris = np.asarray(mesh.triangles)
        mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])
    steps.append(("flipped " + ("".join(a for a, s in zip("xyz", signs) if s < 0) or "none"),
                  o3d.geometry.TriangleMesh(mesh)))

    longest_side = np.max(verts.max(axis=0) - verts.min(axis=0))
    if longest_side > 0:
        mesh.scale(1.0 / longest_side, center=(0, 0, 0))
    steps.append(("scaled", mesh))
    return steps

def normalise_mesh(mesh):
    return normalisation_steps(mesh)[-1][1]


def normalise_all(root, dest):
    os.makedirs(dest, exist_ok=True)
    for path, cls in walk_shapes(root):
        try:
            mesh = o3d.io.read_triangle_mesh(path)
            if np.asarray(mesh.vertices).shape[0] == 0:
                print(f"skipping {path} (empty mesh, 0 verts)")
                continue
            mesh = normalise_mesh(mesh)
            cls_dir = os.path.join(dest, cls)
            os.makedirs(cls_dir, exist_ok=True)
            out_path = os.path.join(cls_dir, os.path.basename(path))
            o3d.io.write_triangle_mesh(out_path, mesh)
        except Exception as e:
            print("couldnt normalise:", path, "-", e)
    print("done normalising : ", dest)

def check_normalisation(root, csv_path=normalisation_csv):
    rows = []
    for path, cls in walk_shapes(root):
        mesh = o3d.io.read_triangle_mesh(path)
        verts = np.asarray(mesh.vertices)
        if verts.shape[0] == 0:
            continue
        center = area_barycenter(mesh)
        eigenvalues, axes = principal_axes(mesh)
        moments = flip_moments(mesh)
        size = verts.max(axis=0) - verts.min(axis=0)
        rows.append({
            "path": path,
            "class": cls,
            "cx": center[0], "cy": center[1], "cz": center[2],
            "barycenter_dist": np.linalg.norm(center),
            "major_x_cos": abs(axes[0, 0]),
            "medium_y_cos": abs(axes[1, 1]),
            "minor_z_cos": abs(axes[2, 2]),
            "moment_x": moments[0], "moment_y": moments[1], "moment_z": moments[2],
            "longest_side": np.max(size),
            "ambiguous_axes": bool(eigenvalues[1] / eigenvalues[0] > 0.99
                                   or eigenvalues[2] / eigenvalues[1] > 0.99),
        })
    df = pd.DataFrame(rows)
    cos = df[["major_x_cos", "medium_y_cos", "minor_z_cos"]]
    moments = df[["moment_x", "moment_y", "moment_z"]]
    off = 1 - cos.min(axis=1)
    print(f"{root}: {len(df)} shapes")
    print(f"  barycenter distance to origin: median {df['barycenter_dist'].median():.2e}, max {df['barycenter_dist'].max():.2e}")
    print(f"  shapes with two (near) equal eigenvalues, so PCA axes are not unique: {df['ambiguous_axes'].sum()}")
    for limit in (1e-6, 1e-3, 1e-2):
        print(f"  shapes with an axis off by more than 1 - |cos| = {limit:g}: {(off > limit).sum()}"
              f" (of which {(off[df['ambiguous_axes']] > limit).sum()} have ambiguous axes)")
    print(f"  shapes with a negative flip moment: {(moments < 0).any(axis=1).sum()}"
          f" (of which {(moments.abs().where(moments < 0) < 1e-3).any(axis=1).sum()} are within 1e-3 of 0)")
    print(f"  longest side: min {df['longest_side'].min():.6g}, max {df['longest_side'].max():.6g}")
    save_stats(df, csv_path)
    return df


def make_normalisation_histograms(before, after):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax = ax.ravel()
    styles = [("before", before, "lightgray", 1.0), ("after", after, None, 0.75)]
    log_bins = lambda lo, hi: np.logspace(lo, hi, 60)
    panels = [
        (ax[0], lambda d: d["barycenter_dist"].clip(1e-16), log_bins(-16, 10), "steelblue",
         "1. translate: barycenter distance to origin", "distance"),
        (ax[1], lambda d: (1 - d["major_x_cos"]).clip(1e-16), log_bins(-16, 0), "seagreen",
         "2. align: 1 - |e1 . x| (major)", "1 - |cos angle| (0 = aligned)"),
        (ax[2], lambda d: (1 - d["medium_y_cos"]).clip(1e-16), log_bins(-16, 0), "seagreen",
         "2. align: 1 - |e2 . y| (medium)", "1 - |cos angle| (0 = aligned)"),
        (ax[3], lambda d: (1 - d["minor_z_cos"]).clip(1e-16), log_bins(-16, 0), "seagreen",
         "2. align: 1 - |e3 . z| for e3 = e1 x e2 (minor)", "1 - |cos angle| (0 = aligned)"),
        (ax[4], lambda d: np.sign(d[["moment_x", "moment_y", "moment_z"]].min(axis=1)), np.linspace(-1.5, 1.5, 4),
         "darkorange", "3. flip: sign of the smallest moment over x, y, z", "sign (+1 = all moments positive)"),
        (ax[5], lambda d: d["longest_side"], log_bins(-2, 10), "indianred",
         "4. scale: longest bounding box side", "length"),
    ]
    for a, value, bins, color, title, xlabel in panels:
        for label, data, fill, alpha in styles:
            a.hist(value(data), bins=bins, color=fill or color, alpha=alpha, edgecolor="black", linewidth=0.3,
                   label=f"{label} normalisation")
        if bins[0] > 0:
            a.set_xscale("log")
        a.set_yscale("log")
        a.set_title(title)
        a.set_xlabel(xlabel)
        a.set_ylabel("# shapes")
        a.legend(fontsize=8)
    ax[4].set_xticks([-1, 0, 1])
    plt.tight_layout()
    fname = os.path.join(out_dir, "histograms_normalised.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved histogram to", fname)


if __name__ == "__main__":
    import sys
    os.makedirs(out_dir, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ("all", "stats", "normalise", "check"):
        sys.exit("usage: python preprocessing.py [all | stats | normalise | check]")
    if mode in ("all", "stats"):
        print("analyzing shapes...")
        stats = make_stats_df(db_path)
        save_stats(stats, stats_csv)
        print_stats_summary(stats)
        make_histograms(stats, label="before")
        make_class_histogram(stats, label="before")
        picked = find_average_and_outliers(stats)
        save_stats(picked, average_outliers_csv)
        if mode == "all":
            print("fixing outliers...")
            fix_outliers(stats, db_path, resampled_dir)
        print("rechecking stats...")
        stats2 = make_stats_df(resampled_dir)
        save_stats(stats2, stats_csv_after)
        print_stats_summary(stats2)
        make_histograms(stats2, label="after")
    if mode in ("all", "normalise"):
        print("normalising...")
        normalise_all(resampled_dir, normalised_dir)
    if mode != "stats":
        before = check_normalisation(resampled_dir, normalisation_before_csv)
        after = check_normalisation(normalised_dir)
        make_normalisation_histograms(before, after)
