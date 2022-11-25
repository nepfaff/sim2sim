from typing import Tuple, Union, List
from abc import ABC, abstractmethod

import numpy as np
from pydrake.all import MultibodyPlant, DiagramBuilder, SceneGraph


class ImageGeneratorBase(ABC):
    """The image generator responsible for placing cameras and taking images."""

    def __init__(self, builder: DiagramBuilder, plant: MultibodyPlant, scene_graph: SceneGraph):
        """
        :param builder: The diagram builder.
        :param plant: The finalized plant.
        :param scene_graph: The scene graph.
        """
        self._builder = builder
        self._plant = plant
        self._scene_graph = scene_graph

    @abstractmethod
    def generate_images(
        self,
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        Union[List[np.ndarray], None],
        Union[List[np.ndarray], None],
    ]:
        """
        Generate images using existing or new cameras.

        :return: A tuple of (images, intrinsics, extrinsics, depths, masks):
            - images: The RGB images of shape (m,n,3) where m is the image height and n is the image width.
            - intrinsics: The intrinsic matrices associated with the images of shape (3,3).
            - extrinsics: The extrinsic matrices associated with the images of shape (4,4).
            - depths: The depth images associated with the images of shape (m,n) where m is the image height and n is
                the image width.
            - labels: The object detection labels. Each image may have multiple associated labels.
            - masks: The binary segmentation masks associated with the labels of type np.uint8 and shape (m,n) where m
                is the image height and n is the image width.
        """
        raise NotImplementedError
