import os
import sys
import numpy as np
import open3d as o3d

from preprocessing import resampled_dir, normalised_dir

# step 3.1: check the full normalisation visually. shows the resampled shape
# before normalising (left) and after translate + align + flip + scale (right),
# each with its own axes: x = red, y = green, z = blue. after normalising the
# shape's longest direction should lie along red, the second longest along green,
# and it should sit in the unit cube (grey). close the window to go to the next one.
# or view specific shapes: python show_normalised.py Vase/m541 Bird/m43
repo_dir = os.path.dirname(os.path.abspath(__file__))

# a mix of long, flat and roundish shapes from different classes, so both
# alignment and flipping get tested on something
default_shapes = ["Biplane/m1120", "Humanoid/m198", "Gun/D00204", "Bird/m43",
                  "Vase/m541", "Door/m1708", "Skyscraper/D00058"]
shapes = sys.argv[1:] if len(sys.argv) > 1 else default_shapes


def axes_at(origin, length=0.5):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=length)
    frame.translate(origin)
    return frame


def unit_cube_at(origin):
    cube = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)))
    cube.paint_uniform_color((0.6, 0.6, 0.6))
    cube.translate(origin)
    return cube


for key in shapes:
    before = o3d.io.read_triangle_mesh(os.path.join(repo_dir, resampled_dir, key + ".obj"))
    after = o3d.io.read_triangle_mesh(os.path.join(repo_dir, normalised_dir, key + ".obj"))
    if len(before.vertices) == 0 or len(after.vertices) == 0:
        print("couldnt load", key)
        continue

    # the resampled shape still has its original position and size, so only
    # scale it down for display (no centering/rotating, that's what we're checking)
    size = np.max(before.get_max_bound() - before.get_min_bound())
    before.scale(1.0 / size, center=before.get_center())
    before.translate((-1.2, 0, 0) - before.get_center())
    after.translate((1.2, 0, 0))

    for mesh in (before, after):
        mesh.compute_vertex_normals()
    title = f"{key} | resampled (left) -> normalised (right) | x red, y green, z blue"
    print("showing", title)
    o3d.visualization.draw_geometries(
        [before, axes_at(before.get_center()), after, axes_at((1.2, 0, 0)), unit_cube_at((1.2, 0, 0))],
        window_name=title, width=1600, height=800, mesh_show_wireframe=True)
