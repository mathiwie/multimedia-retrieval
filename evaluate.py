import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from knn import Engine
from preprocessing import out_dir, save_stats

k = 10
shapes_csv = os.path.join(out_dir, "evaluation_shapes.csv")
classes_csv = os.path.join(out_dir, "evaluation_classes.csv")
metrics = [f"precision@{k}", "r_precision"]


def evaluate(engine):
    labels = engine.database["class"].to_numpy()
    sizes = engine.database["class"].value_counts()
    rows = []
    for i, label in enumerate(labels):
        r = sizes[label] - 1
        found, _ = engine.knn(engine.vectors[i], max(k, r), skip=i)
        same = labels[found] == label
        rows.append({"path": engine.paths[i], "class": label, "class_size": sizes[label],
                     metrics[0]: same[:k].mean(), metrics[1]: same[:r].mean()})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} queries done")
    return pd.DataFrame(rows)


def per_class(shapes):
    n = len(shapes)
    classes = shapes.groupby("class").agg(size=("class_size", "first"),
                                          **{m: (m, "mean") for m in metrics})
    classes["random"] = (classes["size"] - 1) / (n - 1)
    classes["lift"] = classes["r_precision"] / classes["random"]
    return classes.sort_values("r_precision", ascending=False)


def plot(classes, shapes):
    fig, ax = plt.subplots(figsize=(9, 15))
    order = classes.iloc[::-1]
    y = np.arange(len(order))
    ax.barh(y, order["r_precision"], color="steelblue", label="R-precision (K = class size - 1)")
    ax.scatter(order[metrics[0]], y, color="darkorange", s=14, zorder=3, label=f"precision@{k}")
    ax.scatter(order["random"], y, color="black", marker="|", s=60, zorder=3, label="random retrieval")
    for m, style in zip(metrics, ("-", "--")):
        ax.axvline(shapes[m].mean(), color="gray", linestyle=style, linewidth=1,
                   label=f"grand mean {m} = {shapes[m].mean():.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({s})" for c, s in zip(order.index, order["size"])], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("precision")
    ax.set_title("precision per class, every database shape used as query once")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, "evaluation_classes.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved", fname)


def report(shapes, classes):
    print(f"\n{len(shapes)} queries, {len(classes)} classes")
    for m in metrics:
        print(f"  {m:13s} over all queries {shapes[m].mean():.3f}   averaged over classes {classes[m].mean():.3f}")
    print(f"  random retrieval over all queries {((shapes['class_size'] - 1) / (len(shapes) - 1)).mean():.3f}")
    columns = ["size", "r_precision", metrics[0], "random", "lift"]
    print("\nbest classes")
    print(classes[columns].head(10).round(3).to_string())
    print("\nworst classes")
    print(classes[columns].tail(10).round(3).to_string())


def main():
    if os.path.isfile(shapes_csv) and "-redo" not in sys.argv[1:]:
        print("loading", shapes_csv)
        shapes = pd.read_csv(shapes_csv)
    else:
        engine = Engine()
        print(f"querying every shape for its {k} and class size - 1 nearest neighbours...")
        shapes = evaluate(engine)
        save_stats(shapes, shapes_csv)
    classes = per_class(shapes)
    save_stats(classes.reset_index(), classes_csv)
    report(shapes, classes)
    plot(classes, shapes)


if __name__ == "__main__":
    main()
