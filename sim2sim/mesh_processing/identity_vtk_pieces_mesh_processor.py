import os

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult
from .mesh_processor_base import MeshProcessorBase


class IdentityVTKPiecesMeshProcessor(MeshProcessorBase):
    """
    Implements an identity `process_mesh` function reads VTK paths from a directory
    containing VTK files that together make up the processed mesh.
    """

    def __init__(self, logger: DynamicLogger, vtk_pieces_path: str):
        """
        :param mesh_pieces_path: The path to the directory containing the VTK files.
        """
        super().__init__(logger)
        self._vtk_pieces_path = vtk_pieces_path

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> MeshProcessorResult:
        vtk_pieces_paths = []
        with os.scandir(self._vtk_pieces_path) as paths:
            for path in paths:
                if path.is_file() and path.name[-3:].lower() == "vtk":
                    vtk_pieces_paths.append(path.path)

        return MeshProcessorResult(
            result_type=MeshProcessorResult.ResultType.VTK_PATHS,
            vtk_paths=vtk_pieces_paths,
        )
