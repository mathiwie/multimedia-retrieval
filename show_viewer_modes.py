import os
import sys
import open3d as o3d

from preprocessing import db_path, out_dir
from snapshots import render, shaded, save_grid, angled


def main(name="Helicopter/m1304"):
    mesh = shaded(os.path.join(db_path, name + ".obj"))
    lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lines.paint_uniform_color((0.0, 0.0, 0.0))
    view = angled(mesh)
    images = [render(mesh, size=800, view=view), render(mesh, size=800, view=view, wireframe=True),
              render(lines, size=800, view=view)]
    count = f"{len(mesh.vertices):,} v / {len(mesh.triangles):,} f"
    captions = [f"{name}, shaded\n{count}", "shaded with wireframe", "wireframe only"]
    save_grid(images, captions, os.path.join(out_dir, "viewer_modes.png"), columns=3, cell=(3.6, 3.2), fontsize=11)


if __name__ == "__main__":
    main(*sys.argv[1:2])
