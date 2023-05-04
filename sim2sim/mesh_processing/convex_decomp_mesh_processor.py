from typing import Tuple, List, Union, Any, Dict

import trimesh
import open3d as o3d

from .mesh_processor_base import MeshProcessorBase
from sim2sim.util import open3d_to_trimesh
from sim2sim.logging import DynamicLogger


class ConvexDecompMeshProcessor(MeshProcessorBase):
    """Convex decomposition using https://github.com/kmammou/v-hacd."""

    def __init__(
        self, logger: DynamicLogger, mesh_name: str, preview_with_trimesh: bool
    ):
        """
        :param target_sphere_num: The number of spheres that the simplified mesh should contain.
        """
        super().__init__(logger)

        self._mesh_name = mesh_name
        self._preview_with_trimesh = preview_with_trimesh

    def process_mesh(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> Tuple[
        bool,
        Union[o3d.geometry.TriangleMesh, None],
        List[o3d.geometry.TriangleMesh],
        Union[List[Dict[str, Any]], None],
        Union[str, None],
    ]:
        mesh_trimesh = open3d_to_trimesh(mesh)

        if self._preview_with_trimesh:
            scene = trimesh.scene.scene.Scene()
            scene.add_geometry(mesh_trimesh)
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            scene.show()

        try:
            convex_pieces = []
            convex_pieces_new = trimesh.decomposition.convex_decomposition(mesh_trimesh)
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
                f"Showing mesh convex decomp into {len(convex_pieces)} parts. Close window to proceed."
            )
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            scene.show()

        output_meshes = []
        for part in convex_pieces:
            open3d_part = part.as_open3d
            output_meshes.append(open3d_part)

        return False, None, output_meshes, None, None
