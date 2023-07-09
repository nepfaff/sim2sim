from abc import ABC, abstractmethod

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
        self, mesh: o3d.geometry.TriangleMesh
    ) -> PhysicalProperties:
        """
        :param mesh: The mesh to estimate physcial properties for.
        :return: The estimated physical properties.
        """
        raise NotImplementedError
