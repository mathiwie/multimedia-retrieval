import os
import numpy as np
import pandas as pd
import open3d as o3d

from preprocessing import db_path, out_dir, walk_shapes, save_stats, remove_reversed_triangles

check_csv = os.path.join(out_dir, "cleaning_check.csv")


def counts(mesh):
    verts, tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(tris) == 0:
        return len(verts), 0, 0, 0
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    extent = np.max(verts.max(axis=0) - verts.min(axis=0))
    repeated = (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
    return len(verts), len(tris), int(repeated.sum()), int((area <= 1e-12 * extent ** 2).sum())


def clean_steps(path, cls):
    mesh = o3d.io.read_triangle_mesh(path)
    row = {"path": path, "class": cls}
    steps = [("loaded", lambda m: m),
             ("merged", lambda m: m.merge_close_vertices(np.max(m.get_max_bound() - m.get_min_bound()) * 1e-6)),
             ("duplicate vertices", lambda m: m.remove_duplicated_vertices()),
             ("duplicate triangles", lambda m: m.remove_duplicated_triangles()),
             ("reversed triangles", remove_reversed_triangles),
             ("degenerate triangles", lambda m: m.remove_degenerate_triangles()),
             ("unreferenced vertices", lambda m: m.remove_unreferenced_vertices())]
    for name, step in steps:
        if len(mesh.vertices) > 0:
            step(mesh)
        v, f, repeated, flat = counts(mesh)
        row.update({f"{name}: vertices": v, f"{name}: faces": f,
                    f"{name}: repeated corner": repeated, f"{name}: zero area": flat})
    return row


def main():
    data = pd.DataFrame([clean_steps(path, cls) for path, cls in walk_shapes(db_path)])
    save_stats(data, check_csv)
    names = ["loaded", "merged", "duplicate vertices", "duplicate triangles", "reversed triangles",
             "degenerate triangles", "unreferenced vertices"]
    print(f"{len(data)} shapes, totals over the database after each cleaning step")
    print(f"{'step':24s} {'vertices':>10s} {'faces':>10s} {'faces with a repeated corner':>29s} {'zero-area faces':>16s}")
    for name in names:
        print(f"{name:24s} {data[f'{name}: vertices'].sum():>10,} {data[f'{name}: faces'].sum():>10,} "
              f"{data[f'{name}: repeated corner'].sum():>29,} {data[f'{name}: zero area'].sum():>16,}")
    merged = data["loaded: vertices"] - data["merged: vertices"]
    removed = data["loaded: faces"] - data["unreferenced vertices: faces"]
    print(f"\nshapes where vertices were merged: {(merged > 0).sum()}, shapes that lost faces: {(removed > 0).sum()}")
    print(f"most vertices merged in one shape: {merged.max():,} ({data.loc[merged.idxmax(), 'path']})")
    print(f"most faces removed from one shape: {removed.max():,} ({data.loc[removed.idxmax(), 'path']})")
    orphans = data["degenerate triangles: vertices"] - data["unreferenced vertices: vertices"]
    print(f"vertices left without a face by the earlier steps (removed last): {orphans.sum():,} in {(orphans > 0).sum()} shapes")
    reversed_ = data["duplicate triangles: faces"] - data["reversed triangles: faces"]
    print(f"triangles stored a second time with the opposite winding: {reversed_.sum():,} in {(reversed_ > 0).sum()} shapes")
    left = data["unreferenced vertices: zero area"]
    print(f"zero-area faces left after cleaning (three different but collinear corners): {left.sum():,} in {(left > 0).sum()} shapes")


if __name__ == "__main__":
    main()
