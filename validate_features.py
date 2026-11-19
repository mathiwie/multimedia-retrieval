import numpy as np
import open3d as o3d

from features import global_features


def normalise(mesh):
    verts = np.asarray(mesh.vertices)
    verts = verts - verts.mean(axis=0)
    verts = verts / np.max(verts.max(axis=0) - verts.min(axis=0))
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh


def cube(side=1.0):
    return normalise(o3d.geometry.TriangleMesh.create_box(side, side, side
                                                          ).subdivide_midpoint(
                                                              4))


def sphere():
    return normalise(o3d.geometry.TriangleMesh.create_sphere(radius=0.5,
                                                             resolution=60))


def long_box():
    return normalise(o3d.geometry.TriangleMesh.create_box(1.0, 0.1, 0.1
                                                          ).subdivide_midpoint(
                                                              5))


def cylinder():
    return normalise(o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,
                                                               height=1.0,
                                                               resolution=60,
                                                               split=20))


expected = {
    "cube": {"area": 6.0, "compactness": 6 / np.pi, "rectangularity": 1.0,
             "diameter": np.sqrt(3), "convexity": 1.0, "eccentricity": 1.0},
    "sphere": {"area": np.pi, "compactness": 1.0, "rectangularity": np.pi / 6,
               "diameter": 1.0, "convexity": 1.0, "eccentricity": 1.0},
    "long box": {"area": 0.42, "compactness": 0.42 ** 3 / (
        36 * np.pi * 0.01 ** 2),
                 "rectangularity": 1.0, "convexity": 1.0, "diameter": np.sqrt(
                     1 + 2 * 0.1 ** 2),
                 "eccentricity": 56.1},
    "cylinder": {"area": 0.78 * np.pi, "compactness": (0.78 * np.pi) ** 3 / (
        36 * np.pi * (0.09 * np.pi) ** 2),
                 "rectangularity": np.pi / 4, "convexity": 1.0,
                 "diameter": np.sqrt(1 + 0.6 ** 2), "eccentricity": 3.06},
}

shapes = {"cube": cube(), "sphere": sphere(), "long box": long_box(),
          "cylinder": cylinder()}
names = ["area", "compactness", "rectangularity", "diameter", "convexity",
         "eccentricity"]

for name, mesh in shapes.items():
    got = global_features(mesh)
    print(f"\n{name} ({len(mesh.vertices)}v / {len(mesh.triangles)}f)")
    for feature in names:
        want = expected[name].get(feature)
        value = got[feature]
        if want is None:
            print(f"  {feature:15s} {value:10.4f}")
        else:
            error = abs(value - want) / abs(want)
            print(f"  {feature:15s} {value:10.4f}  expected {want:8.4f}",
                  "off by {error:6.2%}"
                  f"  {'ok' if error < 0.05 else 'CHECK'}")
