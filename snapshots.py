import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

fixed_view = {"front": (0.45, -0.75, 0.5
                        ), "lookat": (0.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0
                                                             ), "zoom": 0.6}


def angled(mesh, zoom=0.75):
    return {**fixed_view, "lookat": tuple(
        mesh.get_axis_aligned_bounding_box().get_center()), "zoom": zoom}


def crop(image, pad=10):
    rows, cols = np.nonzero((image < 0.99).any(axis=2))
    if len(rows) == 0:
        return image
    return image[max(rows.min() - pad, 0):rows.max() + pad,
                 max(cols.min() - pad, 0):cols.max() + pad]


def crop_common(images, pad=10):
    mask = np.zeros(images[0].shape[:2], dtype=bool)
    for image in images:
        mask |= (image < 0.99).any(axis=2)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return images
    window = (slice(max(rows.min() - pad, 0), rows.max() + pad),
              slice(max(cols.min() - pad, 0), cols.max() + pad))
    return [image[window] for image in images]


def reference_box(half=1.25):
    box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox((-half,) * 3, (half,) * 3))
    box.paint_uniform_color((1.0, 1.0, 1.0))
    return box


def render(geometries, size=500, view=None, wireframe=False, tight=True):
    if not isinstance(geometries, (list, tuple)):
        geometries = [geometries]
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=size, height=size)
    for geometry in geometries:
        vis.add_geometry(geometry)
    option = vis.get_render_option()
    option.mesh_show_wireframe = wireframe
    option.mesh_show_back_face = True
    if view is not None:
        control = vis.get_view_control()
        control.set_front(view["front"])
        control.set_lookat(view["lookat"])
        control.set_up(view["up"])
        control.set_zoom(view["zoom"])
    vis.poll_events()
    vis.update_renderer()
    image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return crop(image) if tight else image


def shaded(path_or_mesh, color=None):
    mesh = o3d.io.read_triangle_mesh(path_or_mesh) if isinstance(
        path_or_mesh, str) else path_or_mesh
    mesh.compute_vertex_normals()
    if color is not None:
        mesh.paint_uniform_color(color)
    return mesh


def axes_and_cube(size=0.5):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    cube = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
                                            ))
    cube.paint_uniform_color((0.6, 0.6, 0.6))
    return [frame, cube]


def save_grid(images, captions, fname, columns, cell=(2.6, 2.6),
              row_labels=None, fontsize=8, title=None):
    rows = -(-len(images) // columns)
    fig, ax = plt.subplots(rows, columns, figsize=(cell[0] * columns,
                                                   cell[1] * rows
                                                   ), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for i, (image, caption) in enumerate(zip(images, captions)):
        a = ax[i // columns, i % columns]
        a.imshow(image)
        a.set_title(caption, fontsize=fontsize)
    if row_labels:
        for r, label in enumerate(row_labels):
            ax[r, 0].text(-0.08, 0.5, label, transform=ax[r, 0].transAxes,
                          rotation=90,
                          ha="right", va="center", fontsize=fontsize + 1)
    if title:
        fig.suptitle(title, fontsize=fontsize + 2)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close(fig)
    print("saved", fname)
