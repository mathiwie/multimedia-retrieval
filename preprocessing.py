import os
import glob
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
small_thresh = 100
big_thresh = 50000
target_verts = 7500


def get_face_type(path):
    """
    reads the raw .obj text and looks at how many vertex refs each 'f' line
    has, since open3d triangulates everything on load and we'd lose this
    info if we checked mesh.triangles instead. returns 'triangles', 'quads',
    'mixed' or 'none' if the file has no faces at all
    """
    counts = set()
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("f "):
                counts.add(len(line.split()) - 1)
    if not counts:
        return "none"
    if counts == {3}:
        return "triangles"
    if counts == {4}:
        return "quads"
    return "mixed"


def analyze_shape(path, shape_class):
    """
    loads one .obj and grabs everything 2.1 wants:
    class, vertex count, face count, face type, bounding box
    """
    mesh = o3d.io.read_triangle_mesh(path)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    num_verts = verts.shape[0]
    num_faces = tris.shape[0]
    face_type = get_face_type(path)
    if num_verts > 0:
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_min = bbox.get_min_bound()
        bbox_max = bbox.get_max_bound()
    else:
        bbox_min = np.array([np.nan] * 3)
        bbox_max = np.array([np.nan] * 3)
    return {
        "path": path,
        "class": shape_class,
        "num_verts": num_verts,
        "num_faces": num_faces,
        "face_type": face_type,
        "min_x": bbox_min[0], "min_y": bbox_min[1], "min_z": bbox_min[2],
        "max_x": bbox_max[0], "max_y": bbox_max[1], "max_z": bbox_max[2],
    }


def walk_shapes(root):
    """
    yields (path, class_name) for every obj file in the db.
    class name = whatever folder its in, so as an e.g everything under
    ShapeDatabase_INFOMR/AquaticAnimal/ gets labeled AquaticAnimal
    """
    for class_name in sorted(os.listdir(root)):
        class_folder = os.path.join(root, class_name)
        if not os.path.isdir(class_folder):
            continue
        objs = glob.glob(os.path.join(class_folder, "**", "*.obj"), recursive=True)
        for obj_path in objs:
            yield obj_path, class_name


# 2.2: the stats for the whole db
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
    """
    vertex count + face count histograms across the whole db
    using plt.hist.
    """
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
    """
    bar chart of how many shapes per class, so we can eyeball whether
    the (reduced) db is reasonably balanced across classes.
    """
    os.makedirs(out_dir, exist_ok=True)
    counts = df["class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.3), 5))
    ax.bar(counts.index, counts.values, color="seagreen", edgecolor="black")
    ax.set_title("shapes per class " + label)
    ax.set_xlabel("class")
    ax.set_ylabel("# shapes")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"class_histogram_{label.strip().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved class histogram to", fname)


def print_stats_summary(df):
    print("\nsummary ")
    print("num shapes:", len(df))
    print("avg verts:", round(df["num_verts"].mean(), 1))
    print("avg faces:", round(df["num_faces"].mean(), 1))
    print("smallest:", df.loc[df["num_verts"].idxmin(), "path"],
          "-", df["num_verts"].min(), "verts")
    print("biggest:", df.loc[df["num_verts"].idxmax(), "path"],
          "-", df["num_verts"].max(), "verts")
    print("----------------\n")


# 2.3: fixing outliers
def refine_mesh(mesh, target=target_verts):
    """
    for meshes that are too coarse, splits every triangle into 4 by
    adding midpoints on each edge (subdivide_midpoint). each pass roughly
    4x's the vertex count so just loop till we're close to target, capped
    at 6 passes.
    """
    n = np.asarray(mesh.vertices).shape[0]
    count = 0
    while n < target and count < 6:
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        n = np.asarray(mesh.vertices).shape[0]
        count += 1
    return mesh


def simplify_mesh(mesh, target=target_verts):
    """
    for meshes that are too dense. open3d's decimation function wants a
    target triangle count not vertex count so converting using
    the rough euler characteristic relation 
    """
    target_tris = max(int(target * 2), 4)
    return mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)


def fix_outliers(df, root, dest, lo=small_thresh, hi=big_thresh):
    """
    loops through everything in df, refines/simplifies whatever is out of
    range and writes the whole db to dest with the same folder
    layout so walk_shapes still works on it after
    """
    os.makedirs(dest, exist_ok=True)
    for _, row in df.iterrows():
        path = row["path"]
        cls = row["class"]
        nv = row["num_verts"]
        nf = row["num_faces"]
        try:
            if nv == 0:
                print(f"skipping {path} (empty mesh, 0 verts)")
                continue
            mesh = o3d.io.read_triangle_mesh(path)
            too_small = nv < lo or nf < lo
            too_big = nv > hi or nf > hi
            if too_small:
                print(f"refining {path} ({nv}v / {nf}f, too small)")
                mesh = refine_mesh(mesh)
            elif too_big:
                print(f"simplifying {path} ({nv}v / {nf}f, too big)")
                mesh = simplify_mesh(mesh)
            cls_dir = os.path.join(dest, cls)
            os.makedirs(cls_dir, exist_ok=True)
            out_path = os.path.join(cls_dir, os.path.basename(path))
            o3d.io.write_triangle_mesh(out_path, mesh)
        except Exception as e:
            print("couldnt resample:", path, "-", e)
    print("done resampling:", dest)


# 2.5: normalising
def normalise_mesh(mesh):
    """
    move barycenter to origin, then scale so it fits in a unit cube.
    barycenter = mean of vertex positions
    """
    verts = np.asarray(mesh.vertices)
    center = verts.mean(axis=0)
    mesh.translate(-center)
    bbox = mesh.get_axis_aligned_bounding_box()
    size = bbox.get_max_bound() - bbox.get_min_bound()
    longest_side = np.max(size)
    if longest_side > 0:
        mesh.scale(1.0 / longest_side, center=(0, 0, 0))
    return mesh


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


def check_normalisation(root):
    """
    reloads the normalised db and checks barycenter ~ (0,0,0) and
    longest bbox side <= 1 for everything
    """
    rows = []
    for path, cls in walk_shapes(root):
        mesh = o3d.io.read_triangle_mesh(path)
        verts = np.asarray(mesh.vertices)
        if verts.shape[0] == 0:
            continue
        center = verts.mean(axis=0)
        bbox = mesh.get_axis_aligned_bounding_box()
        size = bbox.get_max_bound() - bbox.get_min_bound()
        rows.append({
            "path": path,
            "cx": center[0], "cy": center[1], "cz": center[2],
            "longest_side": np.max(size),
        })
    df = pd.DataFrame(rows)
    print(df.describe())
    return df


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    # 2.1 + 2.2
    print("analyzing shapes...")
    stats = make_stats_df(db_path)
    save_stats(stats, stats_csv)
    print_stats_summary(stats)
    make_histograms(stats, label="before")
    make_class_histogram(stats, label="before")
    # 2.3
    print("fixing outliers...")
    fix_outliers(stats, db_path, resampled_dir)
    # 2.4
    print("rechecking stats...")
    stats2 = make_stats_df(resampled_dir)
    save_stats(stats2, stats_csv_after)
    print_stats_summary(stats2)
    make_histograms(stats2, label="after")
    # 2.5
    print("normalising...")
    normalise_all(resampled_dir, normalised_dir)
    check_normalisation(normalised_dir)