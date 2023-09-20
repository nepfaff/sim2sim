import os

from typing import List

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult

from .mesh_processor_base import MeshProcessorBase


class IdentityVTKPiecesMeshProcessor(MeshProcessorBase):
    """
    Implements an identity `process_mesh` function reads VTK paths from a directory
    containing VTK files that together make up the processed mesh.
    """

    def __init__(self, logger: DynamicLogger, vtk_pieces_paths: List[str]):
        """
        :param vtk_pieces_paths: The paths to the directories containing the VTK files.
        """
        super().__init__(logger)
        self._vtk_pieces_paths = vtk_pieces_paths

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        mesh_processor_results = []
        for dir_path in self._vtk_pieces_paths:
            vtk_pieces_paths = []
            with os.scandir(dir_path) as paths:
                for path in paths:
                    if path.is_file() and path.name[-3:].lower() == "vtk":
                        vtk_pieces_paths.append(path.path)

            mesh_processor_results.append(
                MeshProcessorResult(
                    result_type=MeshProcessorResult.ResultType.VTK_PATHS,
                    vtk_paths=vtk_pieces_paths,
                )
            )

        return mesh_processor_results
