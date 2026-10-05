import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import features as ft
from preprocessing import out_dir, normalised_dir, clean_mesh
from snapshots import render, angled

limit = 1.5
colormap = plt.get_cmap("RdBu_r")


def centred(mesh):
    mesh.translate(-mesh.get_center())
    mesh.scale(1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=(0, 0, 0))
    return mesh


def primitives():
    torus = o3d.geometry.TriangleMesh.create_torus(1.0, 0.3, 120, 50)
    return [("cube", centred(o3d.geometry.TriangleMesh.create_box().subdivide_midpoint(4))),
            ("cylinder", centred(o3d.geometry.TriangleMesh.create_cylinder(0.3, 1.0, 100, 40))),
            ("torus", centred(torus)),
            ("sphere", centred(o3d.geometry.TriangleMesh.create_sphere(0.5, 60)))]


def database(keys):
    return [(key, o3d.io.read_triangle_mesh(os.path.join(normalised_dir, key + ".obj"))) for key in keys]


def resolutions(key):
    mesh = o3d.io.read_triangle_mesh(os.path.join(normalised_dir, key + ".obj"))
    half = clean_mesh(mesh.simplify_quadric_decimation(len(mesh.triangles) // 2))
    return [(f"{key} decimated: {len(half.vertices):,} v", half),
            (f"{key} as stored: {len(mesh.vertices):,} v", mesh),
            (f"{key} subdivided: {len(mesh.subdivide_midpoint(1).vertices):,} v", mesh.subdivide_midpoint(1))]


def coloured(mesh):
    fine, values = ft.curvature_per_vertex(mesh)
    colours = colormap(Normalize(-limit, limit)(np.nan_to_num(values)))[:, :3]
    colours[np.isnan(values)] = (0.6, 0.6, 0.6)
    fine.vertex_colors = o3d.utility.Vector3dVector(colours)
    fine.compute_vertex_normals()
    return fine, values


def main(keys):
    rows = [("primitives", primitives()),
            ("database shapes", database(keys)),
            ("same shape, three resolutions", resolutions(keys[0]))]
    columns = max(len(items) for _, items in rows)
    fig, ax = plt.subplots(len(rows), columns, figsize=(3.4 * columns, 3.4 * len(rows) + 0.6), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for r, (label, items) in enumerate(rows):
        for c, (name, mesh) in enumerate(items):
            fine, values = coloured(mesh)
            ax[r, c].imshow(render(fine, size=700, view=angled(fine, zoom=0.95)))
            ax[r, c].set_title(f"{name}\nmedian {np.nanmedian(values):+.2f}, flat {np.mean(values == 0):.0%}", fontsize=8)
            print(f"{label:30s} {name:40s} median {np.nanmedian(values):+.3f}")
        ax[r, 0].text(-0.08, 0.5, label, transform=ax[r, 0].transAxes, rotation=90, ha="right", va="center")
    bar = fig.colorbar(ScalarMappable(Normalize(-limit, limit), colormap), ax=ax, orientation="horizontal",
                       fraction=0.025, pad=0.02, aspect=60)
    bar.set_label(f"curvature r * kappa (r = {ft.curvature_radius}); blue = concave, white = flat, red = convex")
    fname = os.path.join(out_dir, "curvature_examples.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    main(sys.argv[1:] or ["Humanoid/m198", "Car/D00236", "Mug/D00923", "Bird/m43"])
