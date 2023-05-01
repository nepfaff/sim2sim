from typing import Tuple, List

import open3d as o3d
import numpy as np

from .physical_property_estimator_base import PhysicalPropertyEstimatorBase


class GTPhysicalPropertyEstimator(PhysicalPropertyEstimatorBase):
    """
    Returns the given mass and inertia.
    """

    def __init__(self, mass: float, inertia: List[List[float]]):
        """
        :param mass: The mass in kg.
        :param inertia: The moment of inertia of shape (3,3).
        """
        super().__init__()

        self._mass = mass
        self._inertia = np.asarray(inertia)

    def estimate_physical_properties(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> Tuple[float, np.ndarray]:
        """
        :param mesh: The mesh to estimate physcial properties for.
        :return: A tuple of (mass, inertia).
            - mass: Mass in kg.
            - inertia: Moment of inertia of shape (3,3).
        """
        return self._mass, self._inertia
