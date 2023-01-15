import open3d as o3d

from .mesh_processor_base import MeshProcessorBase
from sim2sim.logging import DynamicLoggerBase


class QuadricDecimationMeshProcessor(MeshProcessorBase):
    """Implements mesh processing through quadric decimation."""

    def __init__(self, logger: DynamicLoggerBase, target_triangle_num: int):
        """
        :param target_triangle_num: The number of triangles that the simplified mesh should contain.
        """
        super().__init__(logger)

        self._target_triangle_num = target_triangle_num

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        :param mesh: The mesh.
        :return: The simplified mesh mesh.
        """
        simplified_mesh = mesh.simplify_quadric_decimation(self._target_triangle_num)
        return simplified_mesh, []
