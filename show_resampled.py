import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d

from preprocessing import stats_csv, stats_csv_after, out_dir, db_path, resampled_dir, save_stats

repo_dir = os.path.dirname(os.path.abspath(__file__))
check_csv = os.path.join(out_dir, "resampling_check.csv")
n = 3

clustered = ["Gun/m660", "House/D00286", "House/D00574", "House/D00901",
             "Humanoid/m198", "Jet/m1176", "RectangleTable/m871", "Vase/m541"]

def load_stats(path):
    df = pd.read_csv(os.path.join(repo_dir, path))
    df["key"] = df["class"] + "/" + df["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    return df.set_index("key")[["class", "num_verts", "num_faces"]]

for path in [stats_csv, stats_csv_after]:
    if not os.path.isfile(os.path.join(repo_dir, path)):
        sys.exit(f"stats not found: {path}, run preprocessing.py first")

stats = load_stats(stats_csv).join(load_stats(stats_csv_after).drop(columns="class"), rsuffix="_after")

if len(sys.argv) > 1:
    picked = stats.loc[sys.argv[1:]].assign(kind="chosen")
else:
    picked = pd.concat([
        stats.nsmallest(n, "num_verts").assign(kind="refined"),
        stats.nlargest(n, "num_verts").assign(kind="simplified"),
        stats.loc[clustered].assign(kind="vertex clustering"),
    ])
    save_stats(picked.reset_index(), os.path.join(repo_dir, check_csv))

def load_for_display(path, offset):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    size = np.max(mesh.get_max_bound() - mesh.get_min_bound())
    mesh.translate(-mesh.get_axis_aligned_bounding_box().get_center())
    if size > 0:
        mesh.scale(1.0 / size, center=(0, 0, 0))
    mesh.translate((offset, 0, 0))
    return mesh

for key, row in picked.iterrows():
    title = (f"{row['kind']}: {key} | original {row['num_verts']}v / {row['num_faces']}f (left)"
             f" -> resampled {row['num_verts_after']}v / {row['num_faces_after']}f (right)")
    print("showing", title)
    original = load_for_display(os.path.join(repo_dir, db_path, key + ".obj"), -0.6)
    resampled = load_for_display(os.path.join(repo_dir, resampled_dir, key + ".obj"), 0.6)
    o3d.visualization.draw_geometries([original, resampled], window_name=title,
                                      width=1600, height=800, mesh_show_wireframe=True)