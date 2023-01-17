from typing import Tuple, List, Union, Any, Dict

import open3d as o3d

from sim2sim.logging import DynamicLoggerBase
from .mesh_processor_base import MeshProcessorBase


class IdentityMeshProcessor(MeshProcessorBase):
    """Implements an identity `process_mesh` function that directly outputs the input mesh."""

    def __init__(self, logger: DynamicLoggerBase):
        super().__init__(logger)

    def process_mesh(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> Tuple[
        bool,
        Union[o3d.geometry.TriangleMesh, None],
        List[o3d.geometry.TriangleMesh],
        Union[List[Dict[str, Any]], None],
    ]:
        return False, mesh, [], None
