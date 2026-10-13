import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.backend_bases import MouseEvent
from sklearn.manifold import TSNE

import distance as ds
import query as qr
from knn import Engine
from preprocessing import out_dir, save_stats

tsne_csv = os.path.join(out_dir, "tsne.csv")
perplexity = 30


def embed_2d(seed=0):
    engine = Engine()
    matrix = ds.distances_to(engine.table, engine.table, engine.scales, engine.weights)
    print(f"running tsne on {len(matrix)} shapes...")
    xy = TSNE(n_components=2, perplexity=perplexity, metric="precomputed", init="random",
              random_state=seed).fit_transform(matrix)
    points = engine.database[["path", "class"]].assign(x=xy[:, 0], y=xy[:, 1])
    save_stats(points, tsne_csv)
    return points


def plot(points, gui=True, brush=None):
    classes = sorted(points["class"].unique())
    code = points["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 10))
    dots = ax.scatter(points["x"], points["y"], c=code, cmap="nipy_spectral", s=10)
    for cls, part in points.groupby("class"):
        ax.text(part["x"].median(), part["y"].median(), cls, fontsize=8, alpha=0.8,
                ha="center", bbox=dict(boxstyle="square,pad=0.1", fc="white", alpha=0.5, lw=0))
    ax.set_title(f"tsne of {len(points)} shapes, {len(classes)} classes")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fname = os.path.join(out_dir, "tsne.png")
    plt.savefig(fname, dpi=150)
    print("saved", fname)

    note = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points", fontsize=9,
                       bbox=dict(boxstyle="round", fc="white"), arrowprops=dict(arrowstyle="-"))
    note.set_visible(False)
    ring = ax.scatter([], [], s=60, facecolors="none", edgecolors="black", linewidths=0.8)

    def picked(event):
        if event.inaxes != ax:
            return None
        hit, info = dots.contains(event)
        return info["ind"][0] if hit else None

    def hover(event):
        i = picked(event)
        if i is None:
            if note.get_visible():
                note.set_visible(False)
                ring.set_offsets(np.empty((0, 2)))
                fig.canvas.draw_idle()
            return
        row = points.iloc[i]
        note.xy = (row["x"], row["y"])
        note.set_text(f"{qr.key(row['path'])}\nclass: {row['class']}")
        note.set_visible(True)
        ring.set_offsets(points.loc[points["class"] == row["class"], ["x", "y"]].to_numpy())
        fig.canvas.draw_idle()

    def click(event):
        i = picked(event)
        if i is None or event.button != 1:
            return
        path = points["path"].iloc[i]
        print("opening", qr.key(path))
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name=qr.key(path), width=900, height=700)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("button_press_event", click)
    if brush is not None:
        row = points[points["path"].map(qr.key) == brush].iloc[0]
        x, y = ax.transData.transform((row["x"], row["y"]))
        fig.canvas.callbacks.process("motion_notify_event", MouseEvent("motion_notify_event", fig.canvas, x, y))
        fname = os.path.join(out_dir, "tsne_brush.png")
        plt.savefig(fname, dpi=150)
        print("saved", fname)
    if gui:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if os.path.isfile(tsne_csv) and "-redo" not in argv:
        print("loading", tsne_csv)
        points = pd.read_csv(tsne_csv)
        points["path"] = points["path"].str.replace("\\", "/", regex=False)
    else:
        points = embed_2d()
    plot(points, gui="-nogui" not in argv, brush=qr.flag_value(argv, "-brush", None, str))
