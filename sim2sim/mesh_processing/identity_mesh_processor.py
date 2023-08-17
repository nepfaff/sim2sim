import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult
from .mesh_processor_base import MeshProcessorBase


class IdentityMeshProcessor(MeshProcessorBase):
    """Implements an identity `process_mesh` function that directly outputs the input mesh."""

    def __init__(self, logger: DynamicLogger):
        super().__init__(logger)

    def process_meshes(self, mesh: o3d.geometry.TriangleMesh) -> MeshProcessorResult:
        return MeshProcessorResult(
            result_type=MeshProcessorResult.ResultType.TRIANGLE_MESH,
            triangle_meshes=[mesh],
        )
