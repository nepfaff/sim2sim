from abc import ABC, abstractmethod
from typing import List

import open3d as o3d

from sim2sim.util import PhysicalProperties


class PhysicalPropertyEstimatorBase(ABC):
    """
    The physical property estimator responsible for estimating the object's physical
    properties.
    """

    def __init__(self):
        pass

    @abstractmethod
    def estimate_physical_properties(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[PhysicalProperties]:
        """
        :param meshes: The meshes to estimate physcial properties for.
        :return: The estimated physical properties.
        """
        raise NotImplementedError
