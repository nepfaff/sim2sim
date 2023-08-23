from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import open3d as o3d


class InverseGraphicsBase(ABC):
    """The inverse graphics responsible for finding object-centric mesh and
    corresponding pose from image data."""

    def __init__(
        self,
        images: List[np.ndarray],
        intrinsics: List[np.ndarray],
        extrinsics: List[np.ndarray],
        depth: Optional[List[np.ndarray]] = None,
        labels: Optional[List[np.ndarray]] = None,
        masks: Optional[List[np.ndarray]] = None,
    ):
        """
        :param images: The RGB images of shape (m,n,3) where m is the image height and n
            is the image width.
        :param intrinsics: The intrinsic matrices associated with the images of shape
            (3,3).
        :param extrinsics: The extrinsic matrices associated with the images of shape
            (4,4).
        :param depths: The depth images associated with the images of shape (m,n) where
            m is the image height and n is the image width.
        :param labels: The object detection labels. Each image may have multiple
            associated labels.
        :param masks: The binary segmentation masks associated with the labels of type
            np.uint8 and shape (m,n) where m is the image height and n is the image
            width.
        """
        self._images = images
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._depth = depth
        self._labels = labels
        self._masks = masks

    @abstractmethod
    def run(self) -> Tuple[List[o3d.geometry.TriangleMesh], np.ndarray]:
        """
        Runs inverse graphics.

        :return: A tuple of (meshes, poses):
            - meshes: The triangular meshes.
            - poses: The poses of form [x, y, z, roll, pitch, yaw] where angles are in
                radians. Has shape (N,6) where N is the number of meshes.
        """
        raise NotImplementedError
