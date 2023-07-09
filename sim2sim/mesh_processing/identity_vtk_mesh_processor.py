import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult
from .mesh_processor_base import MeshProcessorBase


class IdentityVTKMeshProcessor(MeshProcessorBase):
    """
    Implements an identity `process_mesh` function that directly returns the path
    to a given VTK file.
    """

    def __init__(self, logger: DynamicLogger, vtk_path: str):
        super().__init__(logger)
        self._vtk_path = vtk_path

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> MeshProcessorResult:
        return MeshProcessorResult(
            result_type=MeshProcessorResult.ResultType.VTK_PATH,
            vtk_path=self._vtk_path,
        )
