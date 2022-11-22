from abc import ABC, abstractmethod

import open3d as o3d


class MeshProcessorBase(ABC):
    def __init__(self):
        """TODO"""
        pass

    @abstractmethod
    def process_mesh(self, mesh: o3d.geometry.TriangleMesh, **kwargs) -> o3d.geometry.TriangleMesh:
        """TODO"""
        raise NotImplementedError
