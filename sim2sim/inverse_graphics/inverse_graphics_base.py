from typing import Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import open3d as o3d


class InverseGraphicsBase(ABC):
    def __init__(
        self, images: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, depth: Optional[np.ndarray] = None
    ):
        """TODO"""
        self._images = images
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._depth = depth

    @abstractmethod
    def run(self) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """TODO

        :return: A tuple of (mesh, pose):
            -
            -
        """
        raise NotImplementedError
