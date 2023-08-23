from typing import List

import trimesh
import open3d as o3d

from .mesh_processor_base import MeshProcessorBase
from sim2sim.util import open3d_to_trimesh, MeshProcessorResult
from sim2sim.logging import DynamicLogger


class ConvexDecompMeshProcessor(MeshProcessorBase):
    """Convex decomposition using https://github.com/kmammou/v-hacd."""

    def __init__(self, logger: DynamicLogger, preview_with_trimesh: bool):
        super().__init__(logger)

        self._preview_with_trimesh = preview_with_trimesh

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        mesh_processor_results = []
        for mesh in meshes:
            mesh_trimesh = open3d_to_trimesh(mesh)

            if self._preview_with_trimesh:
                scene = trimesh.scene.scene.Scene()
                scene.add_geometry(mesh_trimesh)
                scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
                scene.show()

            try:
                convex_pieces = []
                convex_pieces_new = trimesh.decomposition.convex_decomposition(
                    mesh_trimesh
                )
                if not isinstance(convex_pieces_new, list):
                    convex_pieces_new = [convex_pieces_new]
                convex_pieces += convex_pieces_new
            except Exception as e:
                print(f"Problem performing decomposition: {e}")

            if self._preview_with_trimesh:
                for part in convex_pieces:
                    this_color = trimesh.visual.random_color()
                    part.visual.face_colors[:] = this_color
                scene = trimesh.scene.scene.Scene()
                for part in convex_pieces:
                    scene.add_geometry(part)

                print(
                    f"Showing mesh convex decomp into {len(convex_pieces)} parts. "
                    + "Close window to proceed."
                )
                scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
                scene.show()

            output_meshes = []
            for part in convex_pieces:
                open3d_part = part.as_open3d
                output_meshes.append(open3d_part)

            mesh_processor_results.append(
                MeshProcessorResult(
                    result_type=MeshProcessorResult.ResultType.TRIANGLE_MESH,
                    triangle_meshes=output_meshes,
                )
            )

        return mesh_processor_results
