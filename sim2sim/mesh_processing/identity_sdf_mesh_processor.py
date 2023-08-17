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

    def __init__(self, logger: DynamicLogger, sdf_path: str):
        super().__init__(logger)
        self._sdf_path = sdf_path

    def process_meshes(self, mesh: o3d.geometry.TriangleMesh) -> MeshProcessorResult:
        return MeshProcessorResult(
            result_type=MeshProcessorResult.ResultType.SDF_PATH,
            sdf_path=self._sdf_path,
        )
