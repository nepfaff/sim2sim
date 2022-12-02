from abc import ABC, abstractmethod

import open3d as o3d


class MeshProcessorBase(ABC):
    """
    The mesh processor responsible for postprocessing meshes produced from inverse graphics before adding them to the
    simulation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        :param mesh: The mesh to process.
        :return: The processed mesh.
        """
        raise NotImplementedError
