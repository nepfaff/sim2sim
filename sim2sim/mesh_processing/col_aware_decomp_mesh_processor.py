import logging
import os
from typing import Tuple, List, Union, Any, Dict

import open3d as o3d
import trimesh

from .mesh_processor_base import MeshProcessorBase
from sim2sim.util import open3d_to_trimesh
from sim2sim.logging import DynamicLogger


class CoACDMeshProcessor(MeshProcessorBase):
    """
    Convex decomposition using Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree
    Search [SIGGRAPH2022] from https://github.com/SarahWeiii/CoACD.
    """

    def __init__(self, logger: DynamicLogger, mesh_name: str, mesh_dir: str, preview_with_trimesh: bool):
        """
        :param target_sphere_num: The number of spheres that the simplified mesh should contain.
        :param mesh_dir: The temporary folder to write the convex mesh parts to.
        """
        super().__init__(logger)

        self._mesh_name = mesh_name
        self._mesh_dir = mesh_dir
        self._preview_with_trimesh = preview_with_trimesh

    def process_mesh(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> Tuple[
        bool,
        Union[o3d.geometry.TriangleMesh, None],
        List[o3d.geometry.TriangleMesh],
        Union[List[Dict[str, Any]], None],
    ]:
        """
        Given a mesh, performs a convex decomposition of it with
        coacd, saving all the parts in a subfolder named
        `<mesh_filename>_parts`.

        Args:
        - input_mesh_path: String path to mesh file to decompose. Only
            'obj' format is currently tested, but other formats supported
            by trimesh might work.
        - preview_with_trimesh: Whether to open (and block on) a window to preview
        the decomposition.
        - A set of control kwargs, plus any additional kwargs, are passed to the convex
          decomposition routine 'vhacd'; you can run `testVHACD --help` to see options.

        Returns:
        - List of generated mesh file parts, in obj format.
        """
        # Create a subdir for the convex decomp parts.

        # TODO(liruiw) modify this to temp
        mesh_parts_folder = self._mesh_name + "_parts"
        out_dir = os.path.join(self._mesh_dir, mesh_parts_folder)
        mesh_full_path = self._mesh_name
        os.makedirs(out_dir, exist_ok=True)
        mesh_trimesh = open3d_to_trimesh(mesh)

        if self._preview_with_trimesh:
            logging.info("Showing mesh before decomp. Close window to proceed.")
            scene = trimesh.scene.scene.Scene()
            scene.add_geometry(mesh_trimesh)
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            scene.show()
        try:
            convex_pieces = []
            logging.info("Performing convex decomposition with CoACD.")
            os.system(f"coacd -i {mesh_full_path} -o {out_dir}")

            for file in sorted(os.listdir(out_dir)):
                part = trimesh.load(os.path.join(out_dir, file))
                convex_pieces += [part]

        except Exception as e:
            logging.error(f"Problem performing CoACD decomposition: {e}")

        if self._preview_with_trimesh:
            # Display the convex decomp, giving each a random colors to make them easier to distinguish.
            for part in convex_pieces:
                this_color = trimesh.visual.random_color()
                part.visual.face_colors[:] = this_color
            scene = trimesh.scene.scene.Scene()
            for part in convex_pieces:
                scene.add_geometry(part)

            logging.info(f"Showing mesh convex decomp into {len(convex_pieces)} parts. Close window to proceed.")
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            scene.show()

        os.system(f"rm {out_dir}/*")
        output_meshes = []
        for part in convex_pieces:
            open3d_part = part.as_open3d
            output_meshes.append(open3d_part)
        return False, None, output_meshes, None
