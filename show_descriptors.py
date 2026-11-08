import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d

import features as ft
from snapshots import render, shaded, save_grid, angled
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
    "curvature": "flat or concave vs strongly convex",
}

def caption(row, name, end):
    v = row[name]
    value = f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    return f"{name} {end}: {value}\n{key(row)}"

def key(row):
    return f"{row['class']}/{os.path.splitext(os.path.basename(row['path']))[0]}"

def main(view=False):
    os.makedirs(examples_dir, exist_ok=True)
    data = pd.read_csv(ft.features_csv)
    curv_cols = [f"CURV_{i}" for i in range(ft.n_bins("CURV"))]
    centres = ft.bin_centres("CURV")
    data["curvature"] = (data[curv_cols].values * centres).sum(axis=1)

    picked, images, captions = [], [], []
    for name, story in descriptors.items():
        usable = data[~data.hull_fallback & ~data.curvature_doubtful]
        ordered = usable.sort_values(name).reset_index(drop=True)
        low = ordered.iloc[int(0.02 * len(ordered))]
        high = ordered.iloc[int(0.98 * len(ordered))]
        print(f"{name}: {key(low)} = {low[name]:.3f} (low)  vs  {key(high)} = {high[name]:.3f} (high)   [{story}]")
        pair, labels = [], []
        for row, end in ((low, "low"), (high, "high")):
            mesh = shaded(row["path"], (0.75, 0.75, 0.75))
            pair.append(render(mesh, size=800, view=angled(mesh)))
            labels.append(caption(row, name, end))
            picked.append({"descriptor": name, "end": end, "shape": key(row),
                           "value": row[name], "class": row["class"]})
        save_grid(pair, labels, os.path.join(examples_dir, f"{name}.png"), columns=2, cell=(3.6, 3.6), fontsize=11)
        images += pair
        captions += labels
        if view:
            meshes = [shaded(low["path"]).translate((-0.8, 0, 0)), shaded(high["path"]).translate((0.8, 0, 0))]
            o3d.visualization.draw_geometries(meshes, window_name=f"{name}: {story}", width=1400, height=700)
    save_grid(images, captions, os.path.join(out_dir, "descriptor_examples.png"), columns=4,
              cell=(3.6, 3.4), fontsize=13)

    save_stats(pd.DataFrame(picked), examples_csv)
    print("snapshots in", examples_dir)

if __name__ == "__main__":
    main(view=len(sys.argv) > 1 and sys.argv[1] == "view")