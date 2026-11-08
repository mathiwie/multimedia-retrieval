import os
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

import normalise_features as nf
import query as qr
from knn import Engine
from preprocessing import out_dir, save_stats
from snapshots import render

# the six queries of the step 4 examples (figure 18), the last one as a mesh file from outside the database
queries = ["Gun/D00204", "Humanoid/m198", "Car/D00260", "Bicycle/D00974", "Vase/m541",
           os.path.join("ShapeDatabase_INFOMR", "RoundTable", "m935.obj")]
k = 5
radius = 0.4
examples_csv = os.path.join(out_dir, "knn_examples.csv")


def main():
    engine = Engine()
    rows, grid = [], []
    for target in queries:
        raw, query_path = qr.query_row(target, set(engine.paths))
        q = engine.embed(nf.normalised_table(raw.reset_index(drop=True), engine.params))[0]
        same = np.flatnonzero(engine.paths == query_path.replace("\\", "/"))
        skip = same[0] if len(same) else None
        external = os.path.isfile(target) and not target.startswith(qr.normalised_dir)
        label = qr.key(target) if external else target
        cls = label.split("/")[0]

        found, d = engine.knn(q, k, skip)
        brute = engine.exact(q, np.arange(len(engine.paths)))
        if skip is not None:
            brute[skip] = np.inf
        truth = np.argsort(brute, kind="stable")[:k]
        in_range, _ = engine.range(q, radius, skip)
        labels = engine.database["class"].to_numpy()
        rows.append({"query": label + (" (file)" if external else ""),
                     "knn equals brute force": bool(np.array_equal(found, truth)),
                     "largest distance difference": float(np.abs(d - brute[truth]).max()),
                     f"{k}-nn in class": int((labels[found] == cls).sum()),
                     f"r-nn results (r = {radius})": len(in_range),
                     "r-nn share in class": float((labels[in_range] == cls).mean()) if len(in_range) else np.nan})
        grid.append((label, query_path, external, engine.database.loc[found, "path"].to_list(), d))

    table = pd.DataFrame(rows)
    save_stats(table, examples_csv)
    print(table.round(4).to_string(index=False))

    fig, ax = plt.subplots(len(grid), k + 1, figsize=(2.3 * (k + 1), 2.5 * len(grid)), squeeze=False)
    for r, (label, query_path, external, paths, d) in enumerate(grid):
        for c, path in enumerate([query_path] + paths):
            mesh = o3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color((0.9, 0.5, 0.1) if c == 0 else (0.5, 0.55, 0.65))
            ax[r, c].imshow(render(mesh, size=360))
            ax[r, c].axis("off")
            title = f"query{' (file)' if external else ''}\n{label}" if c == 0 else f"#{c} {qr.key(path)}\nd = {d[c - 1]:.3f}"
            ax[r, c].set_title(title, fontsize=7)
    plt.tight_layout()
    fname = os.path.join(out_dir, "knn_examples.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    main()
