import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

import distance as ds
from snapshots import render
import features as ft
import normalise_features as nf
from preprocessing import (out_dir, normalised_dir, clean_mesh, normalise_mesh,
                           refine_mesh, simplify_mesh, num_verts, target_min, target_max)

normalisation = "log"
histogram_metric = "euclidean"
globals_weight = 6.0
default_k = 5
examples_dir = os.path.join(out_dir, "query_examples")


def prepare_mesh(path):
    mesh = clean_mesh(o3d.io.read_triangle_mesh(path))
    if num_verts(mesh) == 0:
        sys.exit(f"no usable mesh in {path}")
    if num_verts(mesh) < target_min:
        mesh = refine_mesh(mesh)
    elif num_verts(mesh) > target_max:
        mesh = simplify_mesh(mesh)
    return normalise_mesh(mesh)


def query_row(target, database_paths):
    key_path = os.path.join(normalised_dir, target + ".obj")
    saved = pd.read_csv(ft.features_csv)
    match = saved[saved["path"].str.replace("\\", "/", regex=False) == key_path.replace("\\", "/")]
    if len(match) > 0:
        held_out = key_path.replace("\\", "/") not in database_paths
        print(f"query: {target} (features from {ft.features_csv}"
              f"{', held out of the database' if held_out else ''})")
        return match.iloc[0].to_frame().T, key_path
    if not os.path.isfile(target):
        sys.exit(f"not a database shape and not a file: {target}")
    print(f"query: {target} (new mesh, preprocessing it first)")
    prepared = os.path.join(out_dir, "query_shape.obj")
    o3d.io.write_triangle_mesh(prepared, prepare_mesh(target))
    return pd.DataFrame([ft.extract_features(prepared)]), prepared


def show(query_path, matches, label, external=False, gui=True):
    os.makedirs(examples_dir, exist_ok=True)
    captions = [f"query{' (external file)' if external else ''}\n{label}"] + [f"#{rank} {key(row.path)}\nd = {row.distance:.3f}"
                                      for rank, row in enumerate(matches.itertuples(), start=1)]
    meshes = []
    for i, path in enumerate([query_path] + list(matches["path"])):
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color((0.9, 0.5, 0.1) if i == 0 else (0.5, 0.55, 0.65))
        meshes.append(mesh)

    shown = min(len(meshes), 11)
    fig, ax = plt.subplots(1, shown, figsize=(2.4 * shown, 2.9))
    for a, mesh, caption in zip(np.atleast_1d(ax), meshes, captions):
        a.imshow(render(mesh, size=400))
        a.set_title(caption, fontsize=8)
        a.axis("off")
    if shown < len(meshes):
        fig.suptitle(f"first {shown - 1} of {len(meshes) - 1} matches", fontsize=9)
    plt.tight_layout()
    fname = os.path.join(examples_dir, f"{label.replace('/', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("snapshot:", fname)

    if gui:
        for i, mesh in enumerate(meshes):
            mesh.translate((1.1 * i, 0, 0) - mesh.get_center())
        distances = ", ".join(f"{d:.3f}" for d in matches["distance"])
        title = f"{label} -> {len(matches)} best matches (orange = query, left to right: {distances})"
        o3d.visualization.draw_geometries(meshes, window_name=title,
                                          width=min(400 + 300 * len(meshes), 1900), height=600)


def main(target, k=default_k, threshold=None, gui=True):
    everything = pd.read_csv(ft.features_csv)
    database, _ = nf.split_holdout(everything)
    params = nf.fit(database, normalisation)
    database_paths = set(database["path"].str.replace("\\", "/", regex=False))

    raw, query_path = query_row(target, database_paths)
    query = nf.normalised_table(raw.reset_index(drop=True), params)
    table = nf.normalised_table(database.reset_index(drop=True), params)
    print(f"normalised ({normalisation}) global descriptors of the query:")
    print(query[nf.globals_used].iloc[0].astype(float).round(3).to_string())

    weights = {name: (globals_weight if name == "globals" else 1.0) for name in ds.blocks}
    scales = ds.fit_scales(table, histogram_metric)
    distances = ds.distances_to(query, table, scales, weights, histogram_metric)[0]

    results = database.reset_index(drop=True)[["path", "class"]].copy()
    results["distance"] = distances
    results = results[results["path"].str.replace("\\", "/", regex=False)
                      != query_path.replace("\\", "/")]
    results = results.sort_values("distance")
    if threshold is not None:
        results = results[results["distance"] < threshold]
        print(f"{len(results)} shapes closer than {threshold}")
    matches = results.head(k if threshold is None else len(results))

    print(f"\n{'rank':>4}  {'shape':28s} {'class':24s} distance")
    for rank, row in enumerate(matches.itertuples(), start=1):
        print(f"{rank:>4}  {key(row.path):28s} {row._2:24s} {row.distance:.4f}")
    if len(matches) == 0:
        print("  (nothing within that distance)")
        return results

    external = os.path.isfile(target) and not target.startswith(normalised_dir)
    label = key(target) if external else key(query_path)
    show(query_path, matches, label, external, gui)
    return results


def key(path):
    return f"{os.path.basename(os.path.dirname(path))}/{os.path.splitext(os.path.basename(path))[0]}"


def flag_value(argv, flag, default, cast):
    if flag not in argv:
        return default
    i = argv.index(flag)
    if i + 1 >= len(argv):
        sys.exit(f"{flag} needs a value")
    return cast(argv[i + 1])


if __name__ == "__main__":
    argv = sys.argv[1:]
    usage = "usage: python query.py <Class/shape | mesh file> [-k 5] [-t 0.35] [-nogui]"
    if not argv or argv[0].startswith("-"):
        sys.exit(usage)
    k = flag_value(argv, "-k", default_k, int)
    threshold = flag_value(argv, "-t", None, float)
    main(argv[0], k, threshold, gui="-nogui" not in argv)
