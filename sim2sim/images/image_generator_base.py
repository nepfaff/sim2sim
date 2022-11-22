from typing import Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
from pydrake.all import MultibodyPlant


class ImageGeneratorBase(ABC):
    def __init__(self, plant: MultibodyPlant):
        """

        :param plant: Unfinalized plant of the simulation environment.
        """
        self._plant = plant

    @abstractmethod
    def generate_images(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        """

        :return: A tuple of (images, intrinsics, extrinsics, depth):
            -
            -
            -
            -
        """
