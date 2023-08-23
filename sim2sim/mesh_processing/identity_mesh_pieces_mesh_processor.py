import os
from typing import List

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult
from .mesh_processor_base import MeshProcessorBase


class IdentityMeshPiecesMeshProcessor(MeshProcessorBase):
    """
    Implements an identity `process_mesh` function reads mesh pieces from a path and
    returns them.
    """

    def __init__(self, logger: DynamicLogger, mesh_pieces_paths: List[str]):
        """
        :param mesh_pieces_paths: The paths to the directories containing the mesh
            pieces to load.
        """
        super().__init__(logger)
        self._mesh_pieces_paths = mesh_pieces_paths

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        mesh_processor_results = []
        for mesh_pieces_path in self._mesh_pieces_paths:
            mesh_pieces = []
            with os.scandir(mesh_pieces_path) as paths:
                for path in paths:
                    if path.is_file():
                        mesh_piece = o3d.io.read_triangle_mesh(
                            path.path, enable_post_processing=True
                        )
                        mesh_pieces.append(mesh_piece)

            mesh_processor_results.append(
                MeshProcessorResult(
                    result_type=MeshProcessorResult.ResultType.TRIANGLE_MESH,
                    triangle_meshes=mesh_pieces,
                )
            )

        return mesh_processor_results
