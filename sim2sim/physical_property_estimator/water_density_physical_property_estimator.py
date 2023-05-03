from typing import Tuple

import open3d as o3d
import numpy as np

from sim2sim.util import calc_mesh_inertia
from .physical_property_estimator_base import PhysicalPropertyEstimatorBase


class WaterDensityPhysicalPropertyEstimator(PhysicalPropertyEstimatorBase):
    """
    Estimates mass and inertia by assuming a constant density of water.
    """

    def __init__(self):
        super().__init__()

    def estimate_physical_properties(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        :param mesh: The mesh to estimate physcial properties for.
        :return: A tuple of (mass, inertia, center_of_mass).
            - mass: Mass in kg.
            - inertia: Moment of inertia of shape (3,3).
            - center_of_mass: The center of mass that the inertia is about of shape (3,).
        """
        return calc_mesh_inertia(mesh)
