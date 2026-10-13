import os
import random
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import features as ft
from preprocessing import out_dir, normalised_dir, walk_shapes

candidates = [5, 10, 20, 40, 80, 160, 320]
shapes_used = 25


def histograms(values, name, b):
    lo, hi = ft.ranges[name]
    counts, _ = np.histogram(np.clip(values, lo, hi), bins=b, range=(lo, hi))
    return counts / len(values)


def main():
    random.seed(0)
    paths = random.sample([p for p, _ in walk_shapes(normalised_dir)], shapes_used)
    runs, vertex_counts = [], []
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(path)
        vertex_counts.append(len(mesh.vertices))
        runs.append([ft.shape_property_values(mesh, np.random.default_rng(s)) for s in (0, 1)])
    per_vertex_n = min(vertex_counts)
    sample_count = {name: (per_vertex_n if name == "D1" else ft.samples) for name in list(ft.ranges)[:5]}

    names = list(ft.ranges)[:5]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    best, at_bins = {}, {}
    limit = 0.1
    for name in names:
        noise, signal = [], []
        for b in candidates:
            hists = [[histograms(run[name], name, b) for run in shape] for shape in runs]
            noise.append(np.mean([np.abs(a - c).sum() for a, c in hists]))
            pairs = [(i, j) for i in range(len(hists)) for j in range(i + 1, len(hists))]
            signal.append(np.mean([np.abs(hists[i][0] - hists[j][0]).sum() for i, j in pairs]))
        noise, signal = np.array(noise), np.array(signal)
        print(f"\n{name}   B:  " + "  ".join(f"{b:>6d}" for b in candidates))
        print("  noise         " + "  ".join(f"{v:6.3f}" for v in noise))
        print("  signal        " + "  ".join(f"{v:6.3f}" for v in signal))
        print("  noise/signal  " + "  ".join(f"{v:6.1%}" for v in noise / signal))
        at_bins[name] = noise[candidates.index(ft.bins)] / signal[candidates.index(ft.bins)]
        enough = [b for b, s in zip(candidates, signal)
                  if s >= 0.98 * signal.max() and b <= np.sqrt(sample_count[name])]
        best[name] = min(enough) if enough else max(b for b in candidates if b <= np.sqrt(sample_count[name]))
        ax[0].plot(candidates, noise / signal, marker="o", label=name)
        ax[1].plot(candidates, signal, marker="o", label=name)

    ax[0].axhline(limit, color="black", linestyle=":")
    ax[0].set_title(f"noise / signal (want below {limit:.0%})")
    ax[0].set_ylabel("ratio (lower is better)")
    ax[1].set_title("signal: L1 distance between different shapes")
    ax[1].set_ylabel("L1 distance")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("number of bins B")
        a.axvline(np.sqrt(ft.samples), color="black", linestyle="--",
                  label=f"sqrt({ft.samples:,}) = {int(np.sqrt(ft.samples))}")
        a.axvline(np.sqrt(per_vertex_n), color="gray", linestyle="--",
                  label=f"sqrt({per_vertex_n:,}) = {int(np.sqrt(per_vertex_n))}")
        a.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, "bin_choice.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved", fname)
    print("smallest B reaching 98% of the best signal while staying below sqrt(N), per descriptor:", best)
    sampled = {name: ratio for name, ratio in at_bins.items() if name != "D1"}
    print(f"we use a single B = {ft.bins} for all of them: the sampling noise there is at most"
          f" {max(sampled.values()):.1%} of the distance between shapes ({max(sampled, key=sampled.get)}),"
          f" D1 uses every vertex once so it has no sampling noise, and B stays below sqrt(N) for every"
          f" histogram: sqrt({ft.samples:,}) = {int(np.sqrt(ft.samples))} for A3 and D2-D4,"
          f" sqrt({per_vertex_n:,}) = {int(np.sqrt(per_vertex_n))} for D1 (the fewest vertices of any shape here)"
          f" and sqrt({ft.curvature_samples:,}) = {int(np.sqrt(ft.curvature_samples))} for the curvature samples;"
          f" the feature vector stays at {6 * ft.bins} histogram values")


if __name__ == "__main__":
    main()
