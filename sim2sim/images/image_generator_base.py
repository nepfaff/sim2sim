from typing import Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
from pydrake.all import MultibodyPlant


class ImageGeneratorBase(ABC):
    """The image generator responsible for placing cameras and taking images."""

    def __init__(self, plant: MultibodyPlant):
        """
        :param plant: Unfinalized plant of the simulation environment.
        """
        self._plant = plant

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
