import pickle

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult
from .mesh_processor_base import MeshProcessorBase


class IdentityPrimitiveMeshProcessor(MeshProcessorBase):
    """Implements an identity `process_mesh` function that loads primitive info from a pickle file and returns it."""

    def __init__(self, logger: DynamicLogger, primitive_info_path: str):
        super().__init__(logger)

        self._primitive_info_path = primitive_info_path

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> MeshProcessorResult:
        with open(self._primitive_info_path, "rb") as f:
            primitive_info = pickle.load(f)

        return MeshProcessorResult(
            result_type=MeshProcessorResult.ResultType.PRIMITIVE_INFO,
            primitive_info=primitive_info,
        )
