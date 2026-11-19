import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

import features as ft
from preprocessing import out_dir, normalised_dir

spiral_points = 200
hole_angle = 70
# used gemini
vote_colours = ["white", "#fdd0a2", "#fd8d3c", "#9ecae1", "#4292c6", "#08306b"]


def angles(directions):
    between = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1)))
    np.fill_diagonal(between, np.inf)
    random = np.random.default_rng(ft.seed).normal(size=(400_000, 3))
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    nearest = np.degrees(np.arccos(np.clip((random @ directions.T).max(axis=1), -1, 1)))
    to_axis = np.degrees(np.arccos(np.abs(directions).max(axis=1)))
    return between.min(), nearest.max(), np.median(nearest), to_axis.min()


def sphere(ax):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.85", linewidth=0.4)
    for i, name in enumerate("xyz"):
        end = np.zeros(3)
        end[i] = 1.35
        ax.plot(*zip(-end, end), color="0.5", linestyle="--", linewidth=0.8)
        ax.text(*end * 1.12, name, color="0.4", fontsize=9)
    ax.set_box_aspect((1, 1, 1), zoom=1.25)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)


def section(mesh, axis, level):
    corners = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    side = corners[:, :, axis] - level
    cut = (side.min(axis=1) < 0) & (side.max(axis=1) > 0)
    segments = []
    for tri, s in zip(corners[cut], side[cut]):
        points = [tri[a] + s[a] / (s[a] - s[b]) * (tri[b] - tri[a])
                  for a, b in ((0, 1), (1, 2), (2, 0)) if (s[a] < 0) != (s[b] < 0)]
        if len(points) == 2:
            segments.append(points)
    other = [i for i in range(3) if i != axis]
    return np.array(segments)[:, :, other]


def open_sphere(hole=hole_angle):
    # a sphere with diameter 1 without a cap at the top,
    # seen from the centre the hole is a cone of hole degrees
    mesh = o3d.geometry.TriangleMesh.create_sphere(0.5, 80)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    keep = verts[tris].mean(axis=1)[:, 2] < 0.5 * np.cos(np.radians(hole / 2))
    mesh.triangles = o3d.utility.Vector3iVector(tris[keep])
    return mesh.remove_unreferenced_vertices()


def vote_slice(ax, mesh, name, axis):
    grid, _, step = ft.voxel_grid(mesh)
    votes, _ = ft.voxel_votes(mesh)
    count = votes.sum(axis=0).reshape([len(a) for a in grid])
    layer = int(np.argmin(np.abs(grid[axis])))
    plane = np.take(count, layer, axis=axis)
    other = [i for i in range(3) if i != axis]
    extent = [grid[other[0]][0] - step / 2, grid[other[0]][-1] + step / 2,
              grid[other[1]][0] - step / 2, grid[other[1]][-1] + step / 2]
    shown = ax.imshow(plane.T, origin="lower", extent=extent, interpolation="nearest",
                      cmap=ListedColormap(vote_colours), norm=BoundaryNorm(np.arange(-0.5, 6), 6))
    ax.add_collection(LineCollection(section(mesh, axis, grid[axis][layer]), colors="black", linewidths=0.6))
    ax.set_xlabel("xyz"[other[0]])
    ax.set_ylabel("xyz"[other[1]])
    ax.set_aspect("equal")
    leak = ft.leakage(votes)
    possible = plane > 0
    split = possible & (plane < len(votes))
    ax.set_title(f"{name}, slice {'xyz'[axis]} = {round(grid[axis][layer], 2) + 0.0:.2f} through the voxel grid\n"
                 f"leakage of the whole shape L = {leak:.2f}", fontsize=9)
    print(f"{name}: leakage {leak:.3f}, in this slice {split.sum()} of {possible.sum()} possibly inside voxels "
          f"have split votes, {(plane > len(votes) // 2).sum()} are inside by majority")
    return shown


def main(key, axis):
    directions = ft.ray_directions()
    smallest, cover, median, to_axis = angles(directions)
    spiral = ft.golden_spiral(ft.volume_rays)
    print(f"golden spiral with {ft.volume_rays} directions: heights z = {np.round(spiral[:, 2], 2)}, "
          f"turned by {np.degrees(np.pi * (3 - np.sqrt(5))):.1f} degrees each")
    print(f"the {ft.volume_rays} ray directions after the fixed rotation:\n{np.round(directions, 3)}")
    print(f"smallest angle between two rays {smallest:.1f} degrees")
    print(f"every direction lies within {cover:.1f} degrees of a ray (median {median:.1f} degrees)")
    print(f"smallest angle between a ray and a coordinate axis {to_axis:.1f} degrees")

    fig = plt.figure(figsize=(12, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.8, 1), hspace=0.12, wspace=0.15)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    sphere(ax)
    many = ft.golden_spiral(spiral_points)
    ax.scatter(*many.T, c=np.arange(spiral_points), cmap="viridis", s=9)
    ax.set_title(f"(a) golden spiral with {spiral_points} directions (colour = order):\nheights in equal steps, "
                 f"every next one turned by 137.5°", fontsize=9)

    ax = fig.add_subplot(grid[0, 1], projection="3d")
    sphere(ax)
    ax.quiver(0, 0, 0, *directions.T, color="tab:red", linewidth=1.6, arrow_length_ratio=0.12)
    for i, d in enumerate(directions, start=1):
        ax.text(*d * 1.25, str(i), color="tab:red", fontsize=10, fontweight="bold")
    ax.set_title(f"(b) the {ft.volume_rays} ray directions used (spiral of {ft.volume_rays}, fixed rotation)\n"
                 f"at least {smallest:.0f}° apart, every direction within {cover:.0f}° of a ray", fontsize=9)

    slices = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    vote_slice(slices[0], open_sphere(), f"(c) sphere with a hole of {hole_angle}° at the top", 1)
    shown = vote_slice(slices[1], o3d.io.read_triangle_mesh(os.path.join(normalised_dir, key + ".obj")),
                       f"(d) {key}", axis)
    bar = fig.colorbar(shown, ax=slices, ticks=range(6), orientation="horizontal", fraction=0.04, pad=0.1,
                       aspect=50)
    bar.set_label(f"rays that count the voxel as inside (inside if at least {ft.volume_rays // 2 + 1}), "
                  f"black = the surface of the mesh in the slice")
    fname = os.path.join(out_dir, "ray_directions.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv[0] if argv else "Cup/m526", "xyz".index(argv[1]) if len(argv) > 1 else 2)
