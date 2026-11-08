import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

import distance as ds
import features as ft
import query as qr
from compare_distances import feature_scores
from preprocessing import out_dir, normalised_dir, walk_shapes, save_stats

scan_csv = os.path.join(out_dir, "volume_leak.csv")
choice_csv = os.path.join(out_dir, "volume_leak_choice.csv")
limits = {"disagreement": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
          "volume spread": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]}
weights = {name: (qr.globals_weight if name == "globals" else 1.0) for name in ds.blocks}


def occupancy_volume(mesh, res=ft.volume_res):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    verts = np.asarray(mesh.vertices)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    step = (hi - lo).max() / res
    axes = [np.arange(lo[i] + step / 2, hi[i] + step / 2, step) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    return float(scene.compute_occupancy(o3d.core.Tensor(points), nsamples=1).numpy().sum()) * step ** 3


def measure(path, cls):
    mesh = o3d.io.read_triangle_mesh(path)
    votes, cell = ft.voxel_votes(mesh)
    count = votes.sum(axis=0)
    hull, _ = mesh.compute_convex_hull()
    verts, tris = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    row = {"path": path, "class": cls, "watertight": mesh.is_watertight(),
           "area": mesh.get_surface_area(), "hull_volume": ft.closed_mesh_volume(hull),
           "hull_area": hull.get_surface_area(), "obb_volume": mesh.get_oriented_bounding_box().volume(),
           "tetra_volume": np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6,
           "occupancy_volume": occupancy_volume(mesh),
           "voxel_volume": float((count > ft.volume_rays // 2).sum()) * cell,
           "unanimous_volume": float((count == ft.volume_rays).sum()) * cell,
           "any_volume": float((count > 0).sum()) * cell, "leakage": ft.leakage(votes)}
    for i, v in enumerate(votes.sum(axis=1)):
        row[f"direction_{i}"] = float(v) * cell
    return row


def scan():
    rows = []
    for i, (path, cls) in enumerate(walk_shapes(normalised_dir), start=1):
        rows.append(measure(path, cls))
        if i % 250 == 0:
            print(f"  {i} shapes done")
    data = pd.DataFrame(rows)
    save_stats(data, scan_csv)
    return data


def with_volume(features, scan_data, volume, limit):
    out = features.copy()
    for i, r in enumerate(scan_data.itertuples()):
        v, compactness, fallback = ft.settle_volume(volume[i], r.leakage, r.area, r.hull_volume, r.hull_area, limit)
        out.loc[i, ["compactness", "rectangularity", "convexity", "volume", "hull_fallback"]] = \
            [compactness, v / r.obb_volume, v / r.hull_volume, v, fallback]
    return out


def volume_spread(data):
    directions = data[[f"direction_{i}" for i in range(ft.volume_rays)]].to_numpy()
    return (directions.max(axis=1) - directions.min(axis=1)) / data["voxel_volume"].clip(lower=1e-30).to_numpy()


def scores(features):
    return feature_scores(features, weights)


def main(redo=False):
    data = scan() if redo or not os.path.isfile(scan_csv) else pd.read_csv(scan_csv)
    if "occupancy_volume" not in data.columns:
        data.insert(data.columns.get_loc("voxel_volume"), "occupancy_volume",
                    [occupancy_volume(o3d.io.read_triangle_mesh(p)) for p in data["path"]])
        save_stats(data, scan_csv)
    features = pd.read_csv(ft.features_csv)
    key = lambda p: p.str.replace("\\", "/", regex=False)
    assert (key(features["path"]) == key(data["path"])).all()

    tetra = data["tetra_volume"].abs()
    voxel, hull = data["voxel_volume"], data["hull_volume"]
    closed = data["watertight"].astype(bool)
    print(f"{len(data)} shapes, {closed.sum()} watertight")
    print(f"tetrahedra volume: zero for {(tetra < 1e-9).sum()}, negative sum for {(data['tetra_volume'] < 0).sum()}, "
          f"above the hull for {(tetra > hull).sum()}")
    print(f"voxel volume:      zero for {(voxel == 0).sum()}, above the hull for {(voxel > hull).sum()}")
    agree = (tetra[closed] - voxel[closed]).abs() / voxel[closed]
    print(f"watertight shapes: median difference tetrahedra vs voxels {agree.median():.2%}")
    leak = data["leakage"]
    print(f"leakage: watertight median {leak[closed].median():.3f}, 90th pct {leak[closed].quantile(0.9):.3f}; "
          f"all shapes median {leak.median():.3f}, 90th pct {leak.quantile(0.9):.3f}")

    spread = volume_spread(data)
    print(f"volume spread over the {ft.volume_rays} ray directions: watertight median {np.median(spread[closed]):.3f}; "
          f"all shapes median {np.median(spread):.3f}, 90th pct {np.quantile(spread, 0.9):.3f}")

    def row(volume, measure, limit, variant):
        return {"volume": volume, "leak measure": measure, "limit": limit,
                "hull fallbacks": int(variant["hull_fallback"].sum()), **scores(variant)}

    rows = [row("single ray occupancy (old)", "", np.nan,
                with_volume(features, data, data["occupancy_volume"].to_numpy(), 1.0)),
            row("tetrahedra", "", np.nan, with_volume(features, data, tetra.to_numpy(), 1.0)),
            row("convex hull only", "", np.nan, with_volume(features, data, np.full(len(data), np.inf), 1.0)),
            row("voxels, 5 rays", "none", np.nan, with_volume(features, data, voxel.to_numpy(), 1.0))]
    for measure, values in (("disagreement", leak.to_numpy()), ("volume spread", spread)):
        measured = data.assign(leakage=values)
        for limit in limits[measure]:
            rows.append(row("voxels, 5 rays", measure, limit, with_volume(features, measured, voxel.to_numpy(), limit)))
            print(f"  {measure} limit {limit:.2f}: {rows[-1]['hull fallbacks']} fallbacks, "
                  f"p@10 {rows[-1]['precision@10']:.3f}")
    table = pd.DataFrame(rows)
    save_stats(table, choice_csv)
    voxel_rows = table[table["volume"] == "voxels, 5 rays"]
    chosen = voxel_rows.loc[voxel_rows["tune"].idxmax()]
    print("\n" + table.round(3).to_string(index=False))
    print(f"\nchosen on the tune half: leak measure {chosen['leak measure']}, limit {chosen['limit']}, "
          f"test half {chosen['test']:.3f} (features.py: disagreement limit {ft.leak_limit}, which never triggers)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(leak[~closed], bins=40, range=(0, 1), color="indianred", alpha=0.7, label="not watertight")
    ax[0].hist(leak[closed], bins=40, range=(0, 1), color="seagreen", alpha=0.9, label="watertight")
    ax[0].set_xlabel("leakage (share of voxels on which the 5 rays disagree)")
    ax[0].set_ylabel("# shapes")
    ax[0].set_yscale("log")
    ax[0].legend()
    for measure, marker in (("disagreement", "o"), ("volume spread", "s")):
        part = pd.concat([voxel_rows[voxel_rows["leak measure"] == measure], voxel_rows[voxel_rows["leak measure"] == "none"]])
        ax[1].plot(part["hull fallbacks"], part["precision@10"], marker=marker, label=f"leak limit on {measure}")
    for name, style in (("single ray occupancy (old)", ":"), ("tetrahedra", "--"), ("convex hull only", "-.")):
        ax[1].axhline(table.loc[table["volume"] == name, "precision@10"].iloc[0], color="gray", linestyle=style,
                      label=f"{name} (all queries)")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("shapes that use the hull volume (a lower leak limit gives more)")
    ax[1].set_ylabel("precision@10")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, "volume_leak.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    main("-redo" in sys.argv[1:])
