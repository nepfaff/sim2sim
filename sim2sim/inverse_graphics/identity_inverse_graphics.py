from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

from .inverse_graphics_base import InverseGraphicsBase


class IdentityInverseGraphics(InverseGraphicsBase):
    """
    Implements an identity `run` function that loads and directly outputs the specified
    mesh.
    """

    def __init__(
        self,
        mesh_paths: List[str],
        mesh_poses: List[List[float]],
        images: Optional[List[np.ndarray]] = None,
        intrinsics: Optional[List[np.ndarray]] = None,
        extrinsics: Optional[List[np.ndarray]] = None,
        depth: Optional[List[np.ndarray]] = None,
        labels: Optional[List[np.ndarray]] = None,
        masks: Optional[List[np.ndarray]] = None,
    ):
        """
        NOTE: All params apart from `mesh_path` and `mesh_pose` are ignored.
        :param mesh_paths: The paths to load the mesh from.
        :param mesh_poses: The poses of the mesh. Each pose has form
            [roll, pitch, yaw, x, y, z] where angles are in radians.
        """
        super().__init__(images, intrinsics, extrinsics, depth, labels, masks)

        self._meshes = [
            o3d.io.read_triangle_mesh(path, enable_post_processing=True)
            for path in mesh_paths
        ]
        self._mesh_poses = np.asarray(mesh_poses)

    def run(self) -> Tuple[List[o3d.geometry.TriangleMesh], np.ndarray]:
        return self._meshes, self._mesh_poses
