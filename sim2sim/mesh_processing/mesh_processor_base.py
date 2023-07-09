from abc import ABC, abstractmethod

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult


class MeshProcessorBase(ABC):
    """
    The mesh processor responsible for postprocessing meshes produced from inverse
    graphics before adding them to the imulation.
    """

    def __init__(self, logger: DynamicLogger):
        """
        :param logger: The logger.
        """
        self._logger = logger

    @abstractmethod
    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> MeshProcessorResult:
        """
        :param mesh: The mesh to process.
        :return: The mesh processor result.
        """
        raise NotImplementedError
