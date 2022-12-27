from .mesh_processor_base import MeshProcessorBase
import pytorch3d
import pytorch3d.ops as ops
import numpy as np
import torch
import IPython

# https://github.com/mikedh/trimesh/issues/1225
import trimesh
import scipy.spatial
from trimesh import util, convex
import pointnet2_ops.pointnet2_utils as pointnet2_utils
import os


def open3d_to_trimesh(src):
    import open3d as o3d

    """Convert mesh from open3d to trimesh
    https://github.com/wkentaro/morefusion/blob/b8b892b3fbc384982a4929b1418ee29393069b11/morefusion/utils/open3d_to_trimesh.py
    """
    if isinstance(src, o3d.geometry.TriangleMesh):
        vertex_colors = None
        if src.has_vertex_colors:
            vertex_colors = np.asarray(src.vertex_colors)
        dst = trimesh.Trimesh(
            vertices=np.asarray(src.vertices),
            faces=np.asarray(src.triangles),
            vertex_normals=np.asarray(src.vertex_normals),
            vertex_colors=vertex_colors,
        )
    else:
        raise ValueError("Unsupported type of src: {}".format(type(src)))

    return dst


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

    def process_mesh(self, mesh):
        """
        :param mesh: The mesh.
        :return: The simplified mesh mesh.
        """
        mesh_trimesh = open3d_to_trimesh(mesh)

        if self.preview_with_trimesh:  # self.preview_with_trimesh:
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
            print("Problem performing decomposition: %s", e)

        if self.preview_with_trimesh:
            for part in convex_pieces:
                this_color = trimesh.visual.random_color()
                part.visual.face_colors[:] = this_color
            scene = trimesh.scene.scene.Scene()
            for part in convex_pieces:
                scene.add_geometry(part)

            print("Showing mesh convex decomp into %d parts. Close window to proceed." % len(convex_pieces))
            scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
            scene.show()

        output_meshes = []
        for k, part in enumerate(convex_pieces):
            open3d_part = part.as_open3d
            output_meshes.append(open3d_part)

        # convert to open3d geomety
        return None, output_meshes
