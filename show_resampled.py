import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d

from preprocessing import stats_csv, stats_csv_after, out_dir, db_path, resampled_dir, save_stats
from snapshots import render, shaded, save_grid

repo_dir = os.path.dirname(os.path.abspath(__file__))
check_csv = os.path.join(out_dir, "resampling_check.csv")
n = 3

# the shapes that quadric decimation could not bring below 10,000 vertices (output of preprocessing.py)
clustered = ["House/D00286", "House/D00901"]
figure_shapes = [("refined", "AircraftBuoyant/m1337"), ("refined", "Vase/m541"),
                 ("simplified", "Biplane/m1120"), ("simplified", "PlantIndoors/D00080"),
                 ("vertex clustering", "House/D00286"), ("vertex clustering", "House/D00901")]


def load_stats(path):
    df = pd.read_csv(os.path.join(repo_dir, path))
    df["key"] = df["class"] + "/" + df["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    return df.set_index("key")[["class", "num_verts", "num_faces"]]


for path in [stats_csv, stats_csv_after]:
    if not os.path.isfile(os.path.join(repo_dir, path)):
        sys.exit(f"stats not found: {path}, run preprocessing.py first")

stats = load_stats(stats_csv).join(load_stats(stats_csv_after).drop(columns="class"), rsuffix="_after")

args = [a for a in sys.argv[1:] if not a.startswith("-")]
gui = "-nogui" not in sys.argv[1:]
if args:
    picked = stats.loc[args].assign(kind="chosen")
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


def save_figure(stats):
    images, captions = [], []
    for kind, key in figure_shapes:
        row = stats.loc[key]
        for folder, label, nv, nf in ((db_path, "original", row["num_verts"], row["num_faces"]),
                                      (resampled_dir, kind, row["num_verts_after"], row["num_faces_after"])):
            images.append(render(shaded(os.path.join(repo_dir, folder, key + ".obj"), (0.8, 0.8, 0.8)),
                                 size=800, wireframe=True))
            captions.append(f"{key}\n{label}: {nv:,} v / {nf:,} f")
    save_grid(images, captions, os.path.join(repo_dir, out_dir, "resampling_examples.png"), columns=4,
              cell=(3.2, 3.0), fontsize=9)

save_figure(stats)

for key, row in (picked.iterrows() if gui else []):
    title = (f"{row['kind']}: {key} | original {row['num_verts']}v / {row['num_faces']}f (left)"
             f" -> resampled {row['num_verts_after']}v / {row['num_faces_after']}f (right)")
    print("showing", title)
    original = load_for_display(os.path.join(repo_dir, db_path, key + ".obj"), -0.6)
    resampled = load_for_display(os.path.join(repo_dir, resampled_dir, key + ".obj"), 0.6)
    o3d.visualization.draw_geometries([original, resampled], window_name=title,
                                      width=1600, height=800, mesh_show_wireframe=True)