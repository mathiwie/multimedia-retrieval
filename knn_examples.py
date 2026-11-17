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
# the l2 combination is never smaller than the distance of step 4, and r = 0.45 gives the median query
# about as many results (38) as r = 0.4 did with that distance (37)
radius = 0.45
examples_csv = os.path.join(out_dir, "knn_examples.csv")


def main():
    engine = Engine()
    step4 = Engine(combine="sum")
    rows, nearest, in_range = [], [], []
    for target in queries:
        raw, query_path = qr.query_row(target, set(engine.paths))
        normalised = nf.normalised_table(raw.reset_index(drop=True), engine.params)
        q = engine.embed(normalised)[0]
        same = np.flatnonzero(engine.paths == query_path.replace("\\", "/"))
        skip = same[0] if len(same) else None
        external = os.path.isfile(target) and not target.startswith(qr.normalised_dir)
        label = qr.key(target) if external else target
        cls = label.split("/")[0]
        title = f"query{' (file)' if external else ''}\n{label}"

        found, d = engine.knn(q, k, skip)
        brute = engine.exact(q, np.arange(len(engine.paths)))
        if skip is not None:
            brute[skip] = np.inf
        truth = np.argsort(brute, kind="stable")[:k]
        before, _ = step4.knn(step4.embed(normalised)[0], k, skip)
        within, d_within = engine.range(q, radius, skip)
        labels = engine.database["class"].to_numpy()
        hits = int((labels[within] == cls).sum())
        rows.append({"query": label + (" (file)" if external else ""),
                     "knn equals brute force": bool(np.array_equal(found, truth)),
                     "largest distance difference": float(np.abs(d - brute[truth]).max()),
                     f"{k}-nn in class": int((labels[found] == cls).sum()),
                     f"{k}-nn same as step 4": len(set(found) & set(before)),
                     f"r-nn results (r = {radius})": len(within),
                     "r-nn share in class": hits / len(within) if len(within) else np.nan})
        nearest.append((title, query_path, engine.database.loc[found, "path"].to_list(), d))
        in_range.append((f"{title}\n{len(within)} within r = {radius}, {hits} in class", query_path,
                         engine.database.loc[within, "path"].to_list(), d_within))

    table = pd.DataFrame(rows)
    save_stats(table, examples_csv)
    print(table.round(4).to_string(index=False))
    figure(nearest, os.path.join(out_dir, "knn_examples.png"))
    figure(in_range, os.path.join(out_dir, "knn_range_examples.png"))


def figure(grid, fname, shown=k):
    fig, ax = plt.subplots(len(grid), shown + 1, figsize=(2.3 * (shown + 1), 2.6 * len(grid)), squeeze=False)
    for r, (title, query_path, paths, d) in enumerate(grid):
        for a in ax[r]:
            a.axis("off")
        for c, path in enumerate([query_path] + paths[:shown]):
            mesh = o3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color((0.9, 0.5, 0.1) if c == 0 else (0.5, 0.55, 0.65))
            ax[r, c].imshow(render(mesh, size=360))
            ax[r, c].set_title(title if c == 0 else f"#{c} {qr.key(path)}\nd = {d[c - 1]:.3f}", fontsize=7)
        if not paths:
            ax[r, 1].text(0.5, 0.5, "no shape within\nthis distance", ha="center", va="center", fontsize=8,
                          transform=ax[r, 1].transAxes)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    main()
