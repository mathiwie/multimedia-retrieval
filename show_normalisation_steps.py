import os
import sys
import open3d as o3d

from preprocessing import out_dir, resampled_dir, normalisation_steps
from snapshots import render, shaded, axes_and_cube, save_grid, fixed_view, reference_box, crop_common

default_shapes = ["Car/D00236", "Biplane/m1120", "Gun/D00204", "Bird/m43"]


def main(shapes):
    images, captions, labels = [], [], []
    for row, key in enumerate(shapes):
        mesh = o3d.io.read_triangle_mesh(os.path.join(resampled_dir, key + ".obj"))
        if len(mesh.vertices) == 0:
            print("couldnt load", key)
            continue
        labels.append(key)
        for name, state in normalisation_steps(mesh):
            scene = [shaded(state, (0.75, 0.75, 0.75)), reference_box()] + axes_and_cube()
            images.append(render(scene, size=700, view=fixed_view, tight=False))
            captions.append(name if row == 0 else ("flipped " + name.split()[-1] if name.startswith("flipped") else ""))
    save_grid(crop_common(images), captions, os.path.join(out_dir, "normalisation_steps.png"), columns=5,
              cell=(2.8, 2.6), row_labels=labels, fontsize=9)


if __name__ == "__main__":
    main(sys.argv[1:] or default_shapes)
