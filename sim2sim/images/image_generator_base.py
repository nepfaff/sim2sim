from typing import Tuple, Union
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:
        """
        Generate images using existing or new cameras.

        :return: A tuple of (images, intrinsics, extrinsics, depths, masks):
            - images: The images.
            - intrinsics: The intrinsic matrices associated with the images.
            - extrinsics: The extrinsic matrices associated with the images.
            - depths: The depth images associated with the images.
            - masks: The segmentation masks associated with the images. Each image may have multiple associated masks.
        """
        raise NotImplementedError
