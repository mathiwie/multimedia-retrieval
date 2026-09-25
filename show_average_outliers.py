import os
import sys
import pandas as pd
import open3d as o3d

from preprocessing import stats_csv, average_outliers_csv, find_average_and_outliers, save_stats

repo_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(repo_dir, stats_csv)
if not os.path.isfile(csv_path):
    sys.exit(f"stats not found: {csv_path}, run preprocessing.py first")

stats = pd.read_csv(csv_path)
picked = find_average_and_outliers(stats)
save_stats(picked, os.path.join(repo_dir, average_outliers_csv))

for _, row in picked.iterrows():
    title = f"{row['kind']}: {row['class']} - {os.path.basename(row['path'])} ({row['num_verts']}v / {row['num_faces']}f)"
    print("showing", title)
    mesh = o3d.io.read_triangle_mesh(os.path.join(repo_dir, row["path"]))
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name=title, width=1280, height=720, mesh_show_wireframe=True)