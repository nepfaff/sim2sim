from typing import List

import open3d as o3d

from .mesh_processor_base import MeshProcessorBase
from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult


class QuadricDecimationMeshProcessor(MeshProcessorBase):
    """Implements mesh processing through quadric decimation."""

    def __init__(self, logger: DynamicLogger, target_triangle_num: int):
        """
        :param target_triangle_num: The number of triangles that the simplified mesh
            should contain.
        """
        super().__init__(logger)

        self._target_triangle_num = target_triangle_num

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        mesh_processor_results = []
        for mesh in meshes:
            simplified_mesh = mesh.simplify_quadric_decimation(
                self._target_triangle_num
            )
            mesh_processor_results.append(
                MeshProcessorResult(
                    result_type=MeshProcessorResult.ResultType.TRIANGLE_MESH,
                    triangle_meshes=[simplified_mesh],
                )
            )
        return mesh_processor_results
