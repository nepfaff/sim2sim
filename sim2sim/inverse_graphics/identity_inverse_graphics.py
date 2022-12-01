from typing import Tuple, List, Optional

import open3d as o3d
import numpy as np

from .inverse_graphics_base import InverseGraphicsBase


class IdentityInverseGraphics(InverseGraphicsBase):
    """Implements an identity `run` function that loads and directly outputs the specified mesh."""

    def __init__(
        self,
        mesh_path: str,
        mesh_pose: List[float],
        images: Optional[List[np.ndarray]] = None,
        intrinsics: Optional[List[np.ndarray]] = None,
        extrinsics: Optional[List[np.ndarray]] = None,
        depth: Optional[List[np.ndarray]] = None,
        labels: Optional[List[np.ndarray]] = None,
        masks: Optional[List[np.ndarray]] = None,
    ):
        """
        NOTE: All params apart from `mesh_path` and `mesh_pose` are ignored.
        :param mesh_path: The path to load the mesh from.
        :param mesh_pose: The pose of the mesh in form [x, y, z, roll, pitch, yaw] where angles are in radians.
        """
        super().__init__(images, intrinsics, extrinsics, depth, labels, masks)

        self._mesh = o3d.io.read_triangle_mesh(mesh_path)
        self._mesh_pose = np.asarray(mesh_pose)

    def run(self) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """
        Implements an identity function.

        :return: :return: A tuple of (mesh, pose) where pose is all zeros.
        """
        return self._mesh, self._mesh_pose
