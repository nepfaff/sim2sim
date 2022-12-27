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


class ConvexDecompMeshProcessor(MeshProcessorBase):
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
        :param mesh: The mesh.
        :return: The simplified mesh mesh.
        """
        mesh_parts_folder = self.mesh_name + "_parts"
        out_dir = os.path.join(self.mesh_dir, mesh_parts_folder)
        mesh_full_path = os.path.join(self.mesh_dir, self.mesh_name)
        os.makedirs(out_dir, exist_ok=True)

        if preview_with_trimesh:
            logging.info("Showing mesh before decomp. Close window to proceed.")
            scene = trimesh.scene.scene.Scene()
            scene.add_geometry(mesh)
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            scene.show()
        try:
            convex_pieces = []
            logging.info("Performing convex decomposition. If this runs too long, try decreasing --resolution.")
            convex_pieces_new = trimesh.decomposition.convex_decomposition(mesh, **kwargs)
            if not isinstance(convex_pieces_new, list):
                convex_pieces_new = [convex_pieces_new]
            convex_pieces += convex_pieces_new
        except Exception as e:
            logging.error("Problem performing decomposition: %s", e)

        if preview_with_trimesh:
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

        out_paths = []
        for k, part in enumerate(convex_pieces):
            piece_name = "%s_convex_piece_%03d.obj" % (self.mesh_name, k)
            full_path = os.path.join(out_dir, piece_name)
            trimesh.exchange.export.export_mesh(part, full_path)
            out_paths.append(full_path)
        return out_paths
