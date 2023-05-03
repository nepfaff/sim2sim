from typing import Tuple, List

import open3d as o3d
import numpy as np

from .physical_property_estimator_base import PhysicalPropertyEstimatorBase


class GTPhysicalPropertyEstimator(PhysicalPropertyEstimatorBase):
    """
    Returns the given mass and inertia.
    """

    def __init__(
        self, mass: float, inertia: List[List[float]], center_of_mass: List[float]
    ):
        """
        :param mass: The mass in kg.
        :param inertia: The moment of inertia of shape (3,3).
        :param center_of_mass: The center of mass that the inertia is about of shape (3,).
        """
        super().__init__()

        self._mass = mass
        self._inertia = np.asarray(inertia)
        self._center_of_mass = np.asarray(center_of_mass)

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
        return self._mass, self._inertia, self._center_of_mass
