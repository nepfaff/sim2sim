import open3d as o3d
import numpy as np

from typing import Tuple
from inverse_graphics_base import InverseGraphicsBase


class IdentityInverseGraphics(InverseGraphicsBase):
    """Implements an identity `run` function that directly outputs the input inverse graphics."""

    def __init__(self, mesh_path: str):
        super().__init__()
        self._mesh = o3d.io.read_triangle_mesh(mesh_path)

    def run(self) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """
        Implements an identity function.

        :return: :return: A tuple of (mesh, pose) where pose is all zeros.
        """
        return self._mesh, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
