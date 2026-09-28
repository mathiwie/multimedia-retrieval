import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d

import distance as ds
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


def show(query_path, matches, title, gui=True):
    os.makedirs(examples_dir, exist_ok=True)
    meshes = []
    for i, path in enumerate([query_path] + list(matches["path"])):
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        mesh.translate((1.1 * i, 0, 0) - mesh.get_center())
        mesh.paint_uniform_color((0.9, 0.5, 0.1) if i == 0 else (0.75, 0.75, 0.75))
        mesh.compute_vertex_normals()
        meshes.append(mesh)

    width = min(300 + 260 * len(meshes), 1900)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=int(width / (1.15 * len(meshes))) + 120)
    for mesh in meshes:
        vis.add_geometry(mesh)
    vis.get_view_control().set_zoom(0.18)
    vis.poll_events()
    vis.update_renderer()
    name = title.split()[0].replace("/", "_")
    vis.capture_screen_image(os.path.join(examples_dir, f"{name}.png"), do_render=True)
    vis.destroy_window()
    print("snapshot:", os.path.join(examples_dir, f"{name}.png"))
    if gui:
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

    key = lambda p: f"{os.path.basename(os.path.dirname(p))}/{os.path.splitext(os.path.basename(p))[0]}"
    print(f"\n{'rank':>4}  {'shape':28s} {'class':24s} distance")
    for rank, row in enumerate(matches.itertuples(), start=1):
        print(f"{rank:>4}  {key(row.path):28s} {row._2:24s} {row.distance:.4f}")
    if len(matches) == 0:
        print("  (nothing within that distance)")
        return results

    label = target if "/" in target and not os.path.isfile(target) else key(query_path)
    show(query_path, matches, f"{label} -> {len(matches)} best matches (orange = query)", gui)
    return results


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    if not args:
        sys.exit("usage: python query.py <Class/shape | mesh file> [-k 5] [-t 0.35] [-nogui]")
    value = lambda flag, default: (float(args.pop(1)) if flag in flags and len(args) > 1 else default)
    k = int(value("-k", default_k)) if "-k" in flags else default_k
    threshold = value("-t", None) if "-t" in flags else None
    main(args[0], k, threshold, gui="-nogui" not in flags)
