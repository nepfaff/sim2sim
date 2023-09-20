import pickle

from typing import List

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult

from .mesh_processor_base import MeshProcessorBase


class IdentityPrimitiveMeshProcessor(MeshProcessorBase):
    """
    Implements an identity `process_mesh` function that loads primitive info from a
    pickle file and returns it.
    """

    def __init__(self, logger: DynamicLogger, primitive_info_paths: List[str]):
        super().__init__(logger)

        self._primitive_info_paths = primitive_info_paths

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        mesh_processor_results = []
        for path in self._primitive_info_paths:
            with open(path, "rb") as f:
                primitive_info = pickle.load(f)
            mesh_processor_results.append(
                MeshProcessorResult(
                    result_type=MeshProcessorResult.ResultType.PRIMITIVE_INFO,
                    primitive_info=primitive_info,
                )
            )
        return mesh_processor_results
