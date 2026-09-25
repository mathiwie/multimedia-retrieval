import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d

import features as ft
from preprocessing import out_dir, save_stats

examples_dir = os.path.join(out_dir, "descriptor_examples")
examples_csv = os.path.join(out_dir, "descriptor_examples.csv")

descriptors = {
    "area": "small surface vs large crumpled surface",
    "compactness": "sphere-like vs thin and spread out",
    "rectangularity": "round or sparse vs fills its box",
    "diameter": "fits in a ball vs long across the diagonal",
    "convexity": "spiky or hollow vs solid and convex",
    "eccentricity": "bulky vs thin and long",
    "shape_index": "flat or saddle-like vs domes and ridges",
}

def load(path, offset):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.translate((offset, 0, 0))
    return mesh

def snapshot(meshes, fname):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1400, height=700)
    for mesh in meshes:
        vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_wireframe = True
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(fname, do_render=True)
    vis.destroy_window()

def main(view=False):
    os.makedirs(examples_dir, exist_ok=True)
    data = pd.read_csv(ft.features_csv)
    si_cols = [c for c in data.columns if c.startswith("SI_")]
    centres = np.linspace(-1 + 1 / ft.bins, 1 - 1 / ft.bins, ft.bins)
    data["shape_index"] = (data[si_cols].values * centres).sum(axis=1)

    picked = []
    for name, story in descriptors.items():
        usable = data[~data.volume_doubtful & ~data.curvature_doubtful]
        ordered = usable.sort_values(name).reset_index(drop=True)
        low = ordered.iloc[int(0.02 * len(ordered))]
        high = ordered.iloc[int(0.98 * len(ordered))]
        key = lambda row: f"{row['class']}/{os.path.splitext(os.path.basename(row['path']))[0]}"
        title = (f"{name}: {key(low)} = {low[name]:.3f} (left)  vs  "
                 f"{key(high)} = {high[name]:.3f} (right)   [{story}]")
        print(title)
        meshes = [load(low["path"], -0.8), load(high["path"], 0.8)]
        snapshot(meshes, os.path.join(examples_dir, f"{name}.png"))
        if view:
            o3d.visualization.draw_geometries(meshes, window_name=title,
                                              width=1400, height=700, mesh_show_wireframe=True)
        for row, end in ((low, "low"), (high, "high")):
            picked.append({"descriptor": name, "end": end, "shape": key(row),
                           "value": row[name], "class": row["class"]})

    save_stats(pd.DataFrame(picked), examples_csv)
    print("snapshots in", examples_dir)

if __name__ == "__main__":
    main(view=len(sys.argv) > 1 and sys.argv[1] == "view")