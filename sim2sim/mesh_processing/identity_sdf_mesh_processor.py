from typing import List

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult

from .mesh_processor_base import MeshProcessorBase


class IdentitySDFMeshProcessor(MeshProcessorBase):
    """
    Implements an identity `process_mesh` function that directly returns the path
    to a given SDFormat file.
    NOTE: The given SDFormat file must contain the specified 'manipuland_base_link_name'.
    NOTE: Using this mesh processor leads to the SDF file being used directly with
    physical property estimation and other specified properties having no effect.
    """

    def __init__(self, logger: DynamicLogger, sdf_paths: List[str]):
        super().__init__(logger)
        self._sdf_paths = sdf_paths

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        mesh_processor_results = []
        for path in self._sdf_paths:
            mesh_processor_results.append(
                MeshProcessorResult(
                    result_type=MeshProcessorResult.ResultType.SDF_PATH,
                    sdf_path=path,
                )
            )
        return mesh_processor_results
