import open3d as o3d
import numpy as np

from typing import Optional, Tuple
from inverse_graphics_base import InverseGraphicsBase


class IdentityInverseGraphicsProcessor(InverseGraphicsBase):
    """Implements an identity `run` function that directly outputs the input inverse graphics."""

    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh

    def run(self) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """
        Implements an identity function.

        :return: The tuple if the mesh and pose.
        """
        return self.mesh, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
