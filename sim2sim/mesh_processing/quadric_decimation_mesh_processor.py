from typing import Tuple, List, Union, Any, Dict

import open3d as o3d

from .mesh_processor_base import MeshProcessorBase
from sim2sim.logging import DynamicLogger


class QuadricDecimationMeshProcessor(MeshProcessorBase):
    """Implements mesh processing through quadric decimation."""

    def __init__(self, logger: DynamicLogger, target_triangle_num: int):
        """
        :param target_triangle_num: The number of triangles that the simplified mesh should contain.
        """
        super().__init__(logger)

        self._target_triangle_num = target_triangle_num

    def process_mesh(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> Tuple[
        bool,
        Union[o3d.geometry.TriangleMesh, None],
        List[o3d.geometry.TriangleMesh],
        Union[List[Dict[str, Any]], None],
        Union[str, None],
    ]:
        simplified_mesh = mesh.simplify_quadric_decimation(self._target_triangle_num)
        return False, simplified_mesh, [], None, None
