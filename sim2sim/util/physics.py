from typing import Tuple, List

import trimesh
import numpy as np
import open3d as o3d

WATER_DENSITY = 997  # Density of water in kg/m^3


def calc_mesh_inertia(mesh: o3d.geometry.TriangleMesh, density: int = WATER_DENSITY) -> Tuple[float, List[float]]:
    """
    Given a mesh, calculates its total mass and inertia assuming uniform density.

    :param mesh: A trimesh mesh.
    :param density: Density of object in kg/m^3, used for inertia calculation.
    :return: A tuple of (mass, moment_of_inertia).
    """

    mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    mesh_trimesh.density = density
    return mesh_trimesh.mass, mesh_trimesh.moment_inertia
