import open3d as o3d
import numpy as np

from typing import Optional, Tuple
from inverse_graphics_base import InverseGraphicsBase


class IdentityInverseGraphicsProcessor(InverseGraphicsBase):
    """Implements an identity `run` function that directly outputs the input inverse graphics."""

    def __init__(self):
        super().__init__()

    def run(self, mesh) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """
        Implements an identity function.

        :return: The tuple if the mesh and pose.
        """
        return mesh
