import os

from typing import List

import open3d as o3d

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult, convert_obj_to_vtk

from .mesh_processor_base import MeshProcessorBase


class IdentityMeshPiecesMeshProcessor(MeshProcessorBase):
    """
    Implements an identity `process_mesh` function reads mesh pieces from a path and
    returns them.
    """

    def __init__(
        self, logger: DynamicLogger, mesh_pieces_paths: List[str], is_compliant: bool
    ):
        """
        :param mesh_pieces_paths: The paths to the directories containing the mesh
            pieces to load.
        :param is_compliant: Whether to convert the mesh pieces into VTK format to
            simulate as compliant Hydroelastic objects.
        """
        super().__init__(logger)
        self._mesh_pieces_paths = mesh_pieces_paths
        self._is_compliant = is_compliant

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        mesh_processor_results = []
        for i, mesh_pieces_path in enumerate(self._mesh_pieces_paths):
            mesh_pieces = []
            with os.scandir(mesh_pieces_path) as paths:
                for path in paths:
                    if path.is_file():
                        mesh_piece = o3d.io.read_triangle_mesh(
                            path.path, enable_post_processing=True
                        )
                        mesh_pieces.append(mesh_piece)

            if self._is_compliant:
                vtk_pieces_path = os.path.join(
                    self._logger.tmp_dir_path,
                    f"vtk_mesh_pieces_{i}"
                    if len(mesh_pieces) > 1
                    else f"processed_mesh_{i}.vtk",
                )
                if not os.path.exists(vtk_pieces_path):
                    os.mkdir(vtk_pieces_path)
                vtk_paths = convert_obj_to_vtk(
                    obj_meshes=mesh_pieces,
                    output_path=vtk_pieces_path,
                    tmp_folder_path=os.path.join(
                        self._logger.tmp_dir_path, "convert_obj_to_vtk_tmp_folder"
                    ),
                )
                mesh_processor_results.append(
                    MeshProcessorResult(
                        result_type=MeshProcessorResult.ResultType.VTK_PATHS,
                        vtk_paths=vtk_paths,
                    )
                )
            else:
                mesh_processor_results.append(
                    MeshProcessorResult(
                        result_type=MeshProcessorResult.ResultType.TRIANGLE_MESH,
                        triangle_meshes=mesh_pieces,
                    )
                )

        return mesh_processor_results
