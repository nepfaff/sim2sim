from abc import ABC, abstractmethod
from typing import Tuple

import open3d as o3d
import numpy as np


class PhysicalPropertyEstimatorBase(ABC):
    """
    The physical property estimator responsible for estimating the object's mass and inertia.
    NOTE: Should we also estimate friction (e.g. friction between the object and ground)?
    """

    def __init__(self):
        pass

    @abstractmethod
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
        raise NotImplementedError
