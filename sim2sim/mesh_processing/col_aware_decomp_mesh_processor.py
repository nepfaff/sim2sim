import open3d as o3d

from .mesh_processor_base import MeshProcessorBase
import pytorch3d
import pytorch3d.ops as ops
import numpy as np
import torch
import IPython
import trimesh
import scipy.spatial
from trimesh import util, convex
import pointnet2_ops.pointnet2_utils as pointnet2_utils
import logging
import os
from .utils import open3d_to_trimesh


class CoACDMeshProcessor(MeshProcessorBase):
    """Implements mesh processing through quadric decimation."""

    def __init__(self, mesh_name: str, mesh_dir: str, preview_with_trimesh: bool):
        """
        :param target_sphere_num: The number of spheres that the simplified mesh should contain.
        """
        super().__init__()

        # change this to spit out 100 meshes
        self.mesh_name = mesh_name
        self.mesh_dir = mesh_dir
        self.preview_with_trimesh = preview_with_trimesh

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
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
        mesh_parts_folder = self.mesh_name + "_parts"
        out_dir = os.path.join(self.mesh_dir, mesh_parts_folder)
        mesh_full_path = self.mesh_name
        os.makedirs(out_dir, exist_ok=True)
        mesh_trimesh = open3d_to_trimesh(mesh)

        if self.preview_with_trimesh:
            logging.info("Showing mesh before decomp. Close window to proceed.")
            scene = trimesh.scene.scene.Scene()
            scene.add_geometry(mesh_trimesh)
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            # scene.viewer.toggle_axis()
            scene.show()
        try:
            convex_pieces = []
            print(f"coacd -i {mesh_full_path} -o {out_dir}")
            os.system(f"coacd -i {mesh_full_path} -o {out_dir}")
            logging.info("Performing convex decomposition with CoACD.")

            for file in sorted(os.listdir(out_dir)):
                part = trimesh.load(os.path.join(out_dir, file))
                convex_pieces += [part]

        except Exception as e:
            logging.error("Problem performing decomposition: %s", e)

        if self.preview_with_trimesh:
            # Display the convex decomp, giving each a random colors
            # to make them easier to distinguish.
            for part in convex_pieces:
                this_color = trimesh.visual.random_color()
                part.visual.face_colors[:] = this_color
            scene = trimesh.scene.scene.Scene()
            for part in convex_pieces:
                scene.add_geometry(part)

            logging.info("Showing mesh convex decomp into %d parts. Close window to proceed." % (len(convex_pieces)))
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            scene.show()

        # rewrite the mesh with new names
        os.system(f"rm {out_dir}/*")
        output_meshes = []
        for k, part in enumerate(convex_pieces):
            open3d_part = part.as_open3d
            output_meshes.append(open3d_part)
        return None, output_meshes
