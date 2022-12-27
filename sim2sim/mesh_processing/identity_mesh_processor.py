import open3d as o3d

from .mesh_processor_base import MeshProcessorBase


class IdentityMeshProcessor(MeshProcessorBase):
    """Implements an identity `process_mesh` function that directly outputs the input mesh."""

    def __init__(self):
        super().__init__()

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Implements an identity function.

        :param mesh: The mesh.
        :return: The input mesh.
        """
        return mesh, []
