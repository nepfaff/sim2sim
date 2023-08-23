from typing import List

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult
from .mesh_processor_base import MeshProcessorBase


class IdentityMeshProcessor(MeshProcessorBase):
    """
    Implements an identity `process_mesh` function that directly outputs the input mesh.
    """

    def __init__(self, logger: DynamicLogger):
        super().__init__(logger)

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        mesh_processor_results = []
        for mesh in meshes:
            mesh_processor_results.append(
                MeshProcessorResult(
                    result_type=MeshProcessorResult.ResultType.TRIANGLE_MESH,
                    triangle_meshes=[mesh],
                )
            )
        return mesh_processor_results
