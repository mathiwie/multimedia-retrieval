import os
import sys
import open3d as o3d
import numpy as np

repo_dir = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    mesh_path = sys.argv[1]
else:
    mesh_path = os.path.join(repo_dir, "ShapeDatabase_INFOMR", "AquaticAnimal", "m59.obj")

if not os.path.isfile(mesh_path):
    sys.exit(f"mesh not found: {mesh_path}")

mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

print(vertices.shape[0])
print(triangles.shape[0])

vis_option = "wireframe_on_shaded"
if vis_option == "smoothshade":
    o3d.visualization.draw_geometries([mesh], width=1280, height=720)
elif vis_option == "wireframe_on_shaded":
    o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)
elif vis_option == "wireframe":
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh) 
    o3d.visualization.draw_geometries([wireframe], width=1280, height=720)
elif vis_option == "world_axes":
    line_endpoints = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    line_indices = [[0, 1], [0, 2], [0, 3]]

    world_axes = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_endpoints),
        lines=o3d.utility.Vector2iVector(line_indices),
    )

    o3d.visualization.draw_geometries([mesh, world_axes], width=1280, height=720)
elif vis_option == "black_background":
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    vis.run()
    vis.destroy_window()