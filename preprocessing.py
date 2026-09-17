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
average_outliers_csv = os.path.join(out_dir, "average_and_outliers.csv")
normalisation_csv = os.path.join(out_dir, "normalisation_check.csv")
small_thresh = 100
big_thresh = 50000
target_verts = 7500
# 2.3 resamples every shape into this vertex range (assignment: ~5K..10K)
target_min = 5000
target_max = 10000


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


def find_average_and_outliers(df, n=3, lo=small_thresh, hi=big_thresh):
    """
    average shape = the one closest to the mean vertex AND face count
    (each measured in std units so both counts weigh the same).
    outliers = poorly sampled shapes, using the same thresholds as 2.3:
    under lo or over hi verts/faces. returns the average shape plus the
    n smallest and n largest outliers, and how many outliers there are total
    """
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


# 2.3: fixing outliers
def clean_mesh(mesh):
    """
    lots of shapes have the same triangle stored several times (e.g. Bus/D00084
    has 75870 faces but only 25240 unique ones). duplicates stop the decimation
    from collapsing anything and get subdivided too, so remove them first,
    together with duplicate/unused vertices and zero-area triangles.
    vertices that are only a tiny float error apart also count as duplicates,
    otherwise they get merged when the obj is saved and loaded again and the
    vertex count on disk doesn't match (e.g. RectangleTable/m871 5191v -> 4014v)
    """
    if num_verts(mesh) > 0:
        extent = np.max(mesh.get_max_bound() - mesh.get_min_bound())
        mesh.merge_close_vertices(extent * 1e-6)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def num_verts(mesh):
    return np.asarray(mesh.vertices).shape[0]


def search_in_range(simplify, p_min, p_max, target=target_verts, lo=target_min, hi=target_max, iters=12):
    """
    bisection over a simplification parameter p (where a bigger p keeps more
    vertices) until the cleaned result has between lo and hi verts. returns
    the result closest to target if the range can't be hit
    """
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
    """
    for meshes that are too dense. open3d's decimation wants a target triangle
    count, not a vertex count. for a closed mesh faces ~ 2 * verts, so try that
    first. open or messy meshes don't follow that (e.g. Tree/D00765 has as many
    verts as faces), so if it misses, bisect over the triangle target.
    some meshes are so messy (way more faces than verts, e.g. Vase/m541) that
    quadric decimation can't collapse anything at all, for those fall back to
    vertex clustering and bisect over the voxel size instead
    """
    # decimation leaves the collapsed vertices behind, clean so they aren't counted
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


def refine_mesh(mesh, target=target_verts, lo=target_min, hi=target_max):
    """
    for meshes that are too coarse, splits every triangle into 4 by adding
    midpoints on each edge (subdivide_midpoint), so the shape itself doesn't
    change. each pass ~4x's the size, so keep going till we're above lo
    (capped at 8 passes) and if that overshot hi, simplify back down to target.
    subdividing leaves duplicate midpoint vertices behind (they got merged when
    saving, e.g. RectangleTable/m871 5240v in memory but 4014v on disk),
    so clean after every pass
    """
    count = 0
    while num_verts(mesh) < lo and count < 8:
        mesh = clean_mesh(mesh.subdivide_midpoint(number_of_iterations=1))
        count += 1
    if num_verts(mesh) > hi:
        mesh = simplify_mesh(mesh, target, lo, hi)
    return mesh


def fix_outliers(df, root, dest, lo=target_min, hi=target_max):
    """
    loops through everything in df, cleans every mesh, then refines whatever
    has under lo verts and simplifies whatever has over hi verts, so the
    whole db ends up around lo..hi. writes everything to dest with the
    same folder layout so walk_shapes still works on it after
    """
    os.makedirs(dest, exist_ok=True)
    for _, row in df.iterrows():
        path = row["path"]
        cls = row["class"]
        try:
            if row["num_verts"] == 0:
                print(f"skipping {path} (empty mesh, 0 verts)")
                continue
            mesh = clean_mesh(o3d.io.read_triangle_mesh(path))
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


# 2.5 + 3.1: normalising (translate, align, flip, scale)
def principal_axes(verts):
    """
    eigenvectors of the covariance matrix of the vertex positions, as columns,
    sorted from largest to smallest eigenvalue. also returns the eigenvalues
    """
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(verts.T))
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def flip_moments(mesh):
    """
    moment test: for each axis, sum of sign(c) * c^2 over the triangle
    centroids c. positive means most of the mass is on the positive side
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    centroids = verts[tris].mean(axis=1)
    return np.sum(np.sign(centroids) * centroids ** 2, axis=0)


def normalise_mesh(mesh):
    """
    the 4-step normalisation from module 4:
    1. translate: barycenter (mean of vertex positions) to the origin
    2. align: rotate so the largest eigenvector of the covariance matrix lies
       on x and the second largest on y (the smallest ends up on z)
    3. flip: moment test, mirror every axis where most mass is on the negative side
    4. scale: uniformly so the shape tightly fits in a unit cube
    """
    verts = np.asarray(mesh.vertices)
    verts = verts - verts.mean(axis=0)

    _, axes = principal_axes(verts)
    # third axis as the cross product so it's a proper rotation, not a mirror
    rotation = np.stack([axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])])
    verts = verts @ rotation.T
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    signs = np.where(flip_moments(mesh) < 0, -1.0, 1.0)
    verts = verts * signs
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    if np.prod(signs) < 0:
        # an odd number of flips mirrors the shape, which turns the triangles
        # inside out, so swap two corners of each triangle to fix the winding
        tris = np.asarray(mesh.triangles)
        mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])

    longest_side = np.max(verts.max(axis=0) - verts.min(axis=0))
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
    reloads the normalised db and checks all 4 normalisation steps for every shape:
    barycenter ~ (0,0,0), major/medium/minor eigenvector along x/y/z (|cos| ~ 1),
    flip moments all >= 0 and longest bbox side ~ 1. saves the numbers to a
    csv and histograms, so we can show normalisation worked for all shapes
    """
    rows = []
    for path, cls in walk_shapes(root):
        mesh = o3d.io.read_triangle_mesh(path)
        verts = np.asarray(mesh.vertices)
        if verts.shape[0] == 0:
            continue
        center = verts.mean(axis=0)
        _, axes = principal_axes(verts)
        moments = flip_moments(mesh)
        size = verts.max(axis=0) - verts.min(axis=0)
        rows.append({
            "path": path,
            "class": cls,
            "cx": center[0], "cy": center[1], "cz": center[2],
            "barycenter_dist": np.linalg.norm(center),
            # |cos| of the angle between each eigenvector and its axis, 1 = aligned
            "major_x_cos": abs(axes[0, 0]),
            "medium_y_cos": abs(axes[1, 1]),
            "minor_z_cos": abs(axes[2, 2]),
            "moment_x": moments[0], "moment_y": moments[1], "moment_z": moments[2],
            "longest_side": np.max(size),
        })
    df = pd.DataFrame(rows)
    print(df.describe())
    negative = (df[["moment_x", "moment_y", "moment_z"]] < 0).any(axis=1).sum()
    print("shapes with a negative flip moment:", negative)
    save_stats(df, normalisation_csv)
    make_normalisation_histograms(df)
    return df


def make_normalisation_histograms(df):
    """
    one histogram per normalisation step. the x axes are fixed to a sensible
    range around the expected value, otherwise matplotlib zooms in on the
    float rounding noise (~1e-7) and it looks like the shapes are spread out
    when they aren't
    """
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    ax[0].hist(df["barycenter_dist"], bins=50, range=(0, 0.05), color="steelblue", edgecolor="black")
    ax[0].set_title("1. translate: barycenter distance to origin")
    ax[0].set_xlabel("distance (unit cube side = 1)")
    # |cos| can't go above 1, so end the range right at 1 (plus float slack)
    # or rounding splits the fully aligned shapes over two bars
    ax[1].hist(df["major_x_cos"], bins=50, range=(0, 1 + 1e-9), color="seagreen", edgecolor="black")
    ax[1].set_title("2. align: |cos| major eigenvector vs x axis")
    ax[1].set_xlabel("|cos angle| (1 = aligned)")
    smallest_moment = df[["moment_x", "moment_y", "moment_z"]].min(axis=1)
    ax[2].hist(smallest_moment, bins=50, color="darkorange", edgecolor="black")
    ax[2].axvline(0, color="black", linestyle="--")
    ax[2].set_title("3. flip: smallest moment over x, y, z")
    ax[2].set_xlabel("moment (>= 0 = flipped correctly)")
    ax[3].hist(df["longest_side"], bins=51, range=(0.9, 1.1), color="indianred", edgecolor="black")
    ax[3].set_title("4. scale: longest bounding box side")
    ax[3].set_xlabel("length")
    for a in ax:
        a.set_ylabel("# shapes")
        a.ticklabel_format(axis="x", useOffset=False)
    plt.tight_layout()
    fname = os.path.join(out_dir, "histograms_normalised.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved histogram to", fname)


if __name__ == "__main__":
    import sys
    os.makedirs(out_dir, exist_ok=True)
    # "python preprocessing.py normalise" only redoes the normalisation on the
    # already resampled db, so we don't have to wait for 2.1-2.4 again
    only_normalise = len(sys.argv) > 1 and sys.argv[1] == "normalise"
    if not only_normalise:
        # 2.1 + 2.2
        print("analyzing shapes...")
        stats = make_stats_df(db_path)
        save_stats(stats, stats_csv)
        print_stats_summary(stats)
        make_histograms(stats, label="before")
        make_class_histogram(stats, label="before")
        picked = find_average_and_outliers(stats)
        save_stats(picked, average_outliers_csv)
        # 2.3
        print("fixing outliers...")
        fix_outliers(stats, db_path, resampled_dir)
        # 2.4
        print("rechecking stats...")
        stats2 = make_stats_df(resampled_dir)
        save_stats(stats2, stats_csv_after)
        print_stats_summary(stats2)
        make_histograms(stats2, label="after")
    # 2.5 + 3.1
    print("normalising...")
    normalise_all(resampled_dir, normalised_dir)
    check_normalisation(normalised_dir)