from typing import List

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult

from .mesh_processor_base import MeshProcessorBase


class IdentityVTKMeshProcessor(MeshProcessorBase):
    """
    Implements an identity `process_mesh` function that directly returns the path
    to a given VTK file.
    """

    def __init__(self, logger: DynamicLogger, vtk_paths: List[str]):
        super().__init__(logger)
        self._vtk_paths = vtk_paths

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        processed_mesh_results = []
        for path in self._vtk_paths:
            processed_mesh_results.append(
                MeshProcessorResult(
                    result_type=MeshProcessorResult.ResultType.VTK_PATHS,
                    vtk_paths=[path],
                )
            )
        return processed_mesh_results
