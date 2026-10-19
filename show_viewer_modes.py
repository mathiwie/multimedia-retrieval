import os
import sys
import numpy as np
import open3d as o3d

from preprocessing import db_path, out_dir
from snapshots import render, shaded, save_grid, angled

variety = ["Door/D01121", "Spoon/D00422", "Gun/D00204", "Skyscraper/D00058", "Biplane/m1120", "Quadruped/m94"]
moves = [("default camera (as opened)", []),
         ("rotated (mouse drag)", [("rotate", (250.0, -120.0))]),
         ("zoomed in (scroll)", [("rotate", (250.0, -120.0)), ("scale", (-15.0,))]),
         ("panned (ctrl + drag)", [("rotate", (250.0, -120.0)), ("scale", (-15.0,)), ("translate", (120.0, 60.0))])]


def window(mesh, steps=(), size=700, wireframe=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=size, height=size)
    vis.add_geometry(mesh)
    option = vis.get_render_option()
    option.mesh_show_wireframe = wireframe
    option.mesh_show_back_face = True
    control = vis.get_view_control()
    for name, args in steps:
        getattr(control, name)(*args)
    vis.poll_events()
    vis.update_renderer()
    image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return np.pad(image, ((3, 3), (3, 3), (0, 0)), constant_values=0.4)


def describe(name, mesh):
    size = mesh.get_max_bound() - mesh.get_min_bound()
    far = np.abs(np.vstack([mesh.get_min_bound(), mesh.get_max_bound()])).max()
    return f"{name}\n{len(mesh.vertices):,} v / {len(mesh.triangles):,} f\nextent {size.max():.3g}, |coord| up to {far:.3g}"


def main(name="Helicopter/m1304", manipulated="Car/D00236"):
    mesh = shaded(os.path.join(db_path, name + ".obj"))
    lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lines.paint_uniform_color((0.0, 0.0, 0.0))
    view = angled(mesh)
    images = [render(mesh, size=800, view=view), render(mesh, size=800, view=view, wireframe=True),
              render(lines, size=800, view=view)]
    count = f"{len(mesh.vertices):,} v / {len(mesh.triangles):,} f"
    captions = [f"{name}, shaded\n{count}", "shaded with wireframe", "wireframe only"]
    save_grid(images, captions, os.path.join(out_dir, "viewer_modes.png"), columns=3, cell=(3.6, 3.2), fontsize=11)

    mesh = shaded(os.path.join(db_path, manipulated + ".obj"), (0.8, 0.8, 0.8))
    images = [window(mesh, steps) for _, steps in moves]
    captions = [f"{label}\n{manipulated}, {len(mesh.vertices):,} v / {len(mesh.triangles):,} f" for label, _ in moves]
    save_grid(images, captions, os.path.join(out_dir, "viewer_manipulation.png"), columns=4, cell=(3.4, 4.0), fontsize=10)

    meshes = [shaded(os.path.join(db_path, key + ".obj"), (0.8, 0.8, 0.8)) for key in variety]
    save_grid([window(m) for m in meshes], [describe(k, m) for k, m in zip(variety, meshes)],
              os.path.join(out_dir, "viewer_variety.png"), columns=3, cell=(3.6, 4.5), fontsize=10)


if __name__ == "__main__":
    main(*sys.argv[1:3])
