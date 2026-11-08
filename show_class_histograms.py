import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import features as ft
from preprocessing import out_dir, save_stats

default_classes = ["HumanHead", "Bicycle", "Sword", "Car", "Vase", "PlantWildNonTree"]
spread_csv = os.path.join(out_dir, "class_histogram_spread.csv")
titles = {"A3": "A3 (angle)", "D1": "D1 (barycenter to point)", "D2": "D2 (point to point)",
          "D3": "D3 (sqrt triangle area)", "D4": "D4 (cbrt tetrahedron volume)", "CURV": "curvature"}


def histograms(data, name):
    return data[[f"{name}_{i}" for i in range(ft.n_bins(name))]].to_numpy(dtype=float)


def spread(data, sample=600, seed=0):
    labels = data["class"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(data), size=min(sample, len(data)), replace=False)
    same = labels[pick][:, None] == labels[pick][None, :]
    off = ~np.eye(len(pick), dtype=bool)
    rows = []
    for name in ft.ranges:
        h = histograms(data, name)[pick]
        dist = np.abs(h[:, None, :] - h[None, :, :]).sum(-1)
        within, between = dist[same & off].mean(), dist[~same].mean()
        rows.append({"descriptor": name, "within class": within, "between classes": between,
                     "within / between": within / between})
    return pd.DataFrame(rows)


def main(classes):
    data = pd.read_csv(ft.features_csv)
    table = spread(data)
    save_stats(table, spread_csv)
    print("average L1 distance between the histograms of two shapes (600 random shapes)")
    print(table.round(3).to_string(index=False))

    names = list(ft.ranges)
    fig, ax = plt.subplots(len(classes), len(names), figsize=(2.9 * len(names), 2.1 * len(classes)),
                           sharex="col", sharey="col", squeeze=False)
    colors = plt.cm.tab10(np.arange(len(names)))
    for r, cls in enumerate(classes):
        part = data[data["class"] == cls]
        for c, name in enumerate(names):
            x = ft.bin_centres(name)
            h = histograms(part, name)
            a = ax[r, c]
            a.plot(x, h.T, color=colors[c], alpha=0.25, linewidth=0.7)
            a.plot(x, h.mean(axis=0), color="black", linewidth=1.4)
            if r == 0:
                a.set_title(titles[name], fontsize=10)
            if c == 0:
                a.set_ylabel(f"{cls}\n({len(part)} shapes)", fontsize=9)
            a.tick_params(labelsize=7)
    fig.suptitle("every shape of a class as one line (black: class mean)", fontsize=11)
    plt.tight_layout()
    fname = os.path.join(out_dir, "class_histograms.png")
    plt.savefig(fname, dpi=170)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    main(sys.argv[1:] or default_classes)
