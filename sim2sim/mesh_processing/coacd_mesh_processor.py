import logging
import time
import os

import numpy as np
import open3d as o3d
import trimesh
import coacd

from .mesh_processor_base import MeshProcessorBase
from sim2sim.util import open3d_to_trimesh, MeshProcessorResult, convert_obj_to_vtk
from sim2sim.logging import DynamicLogger


class CoACDMeshProcessor(MeshProcessorBase):
    """
    Convex decomposition using Approximate Convex Decomposition for 3D Meshes with
    Collision-Aware Concavity and Tree Search [SIGGRAPH2022] from
    https://github.com/SarahWeiii/CoACD.
    """

    def __init__(
        self,
        logger: DynamicLogger,
        threshold: float,
        resolution: int,
        preprocess_resolution: int,
        mcts_nodes: int,
        mcts_iterations: int,
        mcts_max_depth: int,
        preview_with_trimesh: bool,
        is_compliant: bool,
    ):
        """
        :param threshold: Termination criteria in [0.01, 1] (0.01: most fine-grained;
            1: most coarse).
        :param resolution: Surface samping resolution for Hausdorff distance computation.
        :param preprocess_resolution: Preprocessing resolution.
        :param mcts_nodes: Number of cut candidates for MCTS.
        :param mcts_iterations: Number of MCTS iterations.
        :param mcts_max_depth: Maximum depth for MCTS search.
        :param preview_with_trimesh: Whether to visualize the original and CoACD mesh
            in a trimesh popup.
        :param is_compliant: Whether to convert the mesh pieces into VTK format to
            simulate as compliant Hydroelastic objects.
        """
        super().__init__(logger)

        self._threshold = threshold
        self._preprocess_resolution = preprocess_resolution
        self._resolution = resolution
        self._mcts_nodes = mcts_nodes
        self._mcts_iterations = mcts_iterations
        self._mcts_max_depth = mcts_max_depth
        self._preview_with_trimesh = preview_with_trimesh
        self._is_compliant = is_compliant

        # Prevent info logs
        coacd.set_log_level("error")

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> MeshProcessorResult:
        if self._preview_with_trimesh:
            logging.info("Showing mesh before decomp. Close window to proceed.")
            mesh_trimesh = open3d_to_trimesh(mesh)
            scene = trimesh.scene.scene.Scene()
            scene.add_geometry(mesh_trimesh)
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            scene.show()

        # Perform CoACD convex decomposition
        coacd_mesh = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        start_time = time.time()
        coacd_result = coacd.run_coacd(
            coacd_mesh,
            threshold=self._threshold,
            preprocess_resolution=self._preprocess_resolution,
            resolution=self._resolution,
            mcts_nodes=self._mcts_nodes,
            mcts_iterations=self._mcts_iterations,
            mcts_max_depth=self._mcts_max_depth,
        )
        coacd_duration_s = time.time() - start_time
        self._logger.log(meta_data={"coacd_duration_s": coacd_duration_s})

        # Extract CoACD result
        convex_pieces = []
        for vs, fs in coacd_result:
            convex_pieces.append(trimesh.Trimesh(vs, fs))

        if self._preview_with_trimesh:
            # Display the convex decomp, giving each a random colors to make them easier
            # to distinguish.
            for part in convex_pieces:
                this_color = trimesh.visual.random_color()
                part.visual.face_colors[:] = this_color
            scene = trimesh.scene.scene.Scene()
            for part in convex_pieces:
                scene.add_geometry(part)

            logging.info(
                f"Showing mesh convex decomp into {len(convex_pieces)} parts. Close "
                + "window to proceed."
            )
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            scene.show()

        # Convert meshes from trimesh to open3d
        output_meshes = []
        for part in convex_pieces:
            open3d_part = part.as_open3d
            output_meshes.append(open3d_part)

        if self._is_compliant:
            vtk_pieces_path = os.path.join(
                self._logger.tmp_dir_path,
                "vtk_mesh_pieces" if len(output_meshes) > 1 else "processed_mesh.vtk",
            )
            if not os.path.exists(vtk_pieces_path):
                os.mkdir(vtk_pieces_path)
            vtk_paths = convert_obj_to_vtk(
                obj_meshes=output_meshes,
                output_path=vtk_pieces_path,
                tmp_folder_path=os.path.join(
                    self._logger.tmp_dir_path, "convert_obj_to_vtk_tmp_folder"
                ),
            )
            return MeshProcessorResult(
                result_type=MeshProcessorResult.ResultType.VTK_PATHS,
                vtk_paths=vtk_paths,
            )

        return MeshProcessorResult(
            result_type=MeshProcessorResult.ResultType.TRIANGLE_MESH,
            triangle_meshes=output_meshes,
        )
