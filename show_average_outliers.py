import os
import sys
import pandas as pd
import open3d as o3d

from preprocessing import stats_csv, average_outliers_csv, out_dir, find_average_and_outliers, save_stats
from snapshots import render, shaded, save_grid, angled

repo_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(repo_dir, stats_csv)
if not os.path.isfile(csv_path):
    sys.exit(f"stats not found: {csv_path}, run preprocessing.py first")

stats = pd.read_csv(csv_path)
picked = find_average_and_outliers(stats)
save_stats(picked, os.path.join(repo_dir, average_outliers_csv))

images, captions = [], []
for _, row in picked.iterrows():
    path = row["path"].replace("\\", "/")
    key = f"{row['class']}/{os.path.splitext(os.path.basename(path))[0]}"
    mesh = shaded(os.path.join(repo_dir, path), (0.8, 0.8, 0.8))
    images.append(render(mesh, size=800, view=angled(mesh), wireframe=True))
    captions.append(f"{row['kind']}: {key}\n{row['num_verts']:,} v / {row['num_faces']:,} f")
save_grid(images, captions, os.path.join(repo_dir, out_dir, "average_and_outliers.png"), columns=4,
          cell=(3.2, 3.0), fontsize=9)

if "-nogui" not in sys.argv[1:]:
    for _, row in picked.iterrows():
        path = row["path"].replace("\\", "/")
        title = f"{row['kind']}: {row['class']} - {os.path.basename(path)} ({row['num_verts']}v / {row['num_faces']}f)"
        print("showing", title)
        mesh = o3d.io.read_triangle_mesh(os.path.join(repo_dir, path))
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name=title, width=1280, height=720, mesh_show_wireframe=True)
