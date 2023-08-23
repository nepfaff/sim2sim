from abc import ABC, abstractmethod
from typing import List

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
    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        """
        :param meshes: The meshes to process.
        :return: The mesh processor results.
        """
        raise NotImplementedError
