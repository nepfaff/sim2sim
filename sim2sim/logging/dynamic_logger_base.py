from abc import ABC, abstractmethod
from typing import Union, Optional, List
import os
import datetime

import numpy as np
from pydrake.all import MultibodyPlant


class DynamicLoggerBase(ABC):
    """Dynamics logger base class."""

    def __init__(self, logging_frequency_hz: float, logging_path: str):
        """
        :param logging_frequency_hz: The frequency at which we want to log at.
        :param logging_path: The path to the directory that we want to write the log files to.
        """
        self._logging_frequency_hz = logging_frequency_hz
        self._logging_path = logging_path

        self._outer_plant: Union[MultibodyPlant, None] = None
        self._inner_plant: Union[MultibodyPlant, None] = None

        if not os.path.exists(logging_path):
            os.mkdir(logging_path)

        self._creation_timestamp = str(datetime.datetime.now())

        # Data directory names in `logging_path`
        self._camera_poses_dir_path = os.path.join(logging_path, "camera_poses")
        self._intrinsics_dir_path = os.path.join(logging_path, "intrinsics")
        self._images_dir_path = os.path.join(logging_path, "images")
        self._depths_dir_path = os.path.join(logging_path, "depths")
        self._masks_dir_path = os.path.join(logging_path, "binary_masks")
        self._data_directory_paths = [
            self._camera_poses_dir_path,
            self._intrinsics_dir_path,
            self._images_dir_path,
            self._depths_dir_path,
            self._masks_dir_path,
        ]
        self._meta_data_file_path = os.path.join(logging_path, "meta_data.yaml")

        # Logging data
        self._camera_poses: Optional[List[np.ndarray]] = []
        self._intrinsics: Optional[List[np.ndarray]] = []
        self._images: Optional[List[np.ndarray]] = []
        self._depths: Optional[List[np.ndarray]] = []
        self._labels: Optional[List[np.ndarray]] = []
        self._masks: Optional[List[np.ndarray]] = []

    def add_plants(self, outer_plant: MultibodyPlant, inner_plant: MultibodyPlant) -> None:
        """Add finalized plants."""
        self._outer_plant = outer_plant
        self._inner_plant = inner_plant

    @abstractmethod
    def log(
        self,
        camera_poses: Optional[List[np.ndarray]],
        intrinsics: Optional[List[np.ndarray]],
        images: Optional[List[np.ndarray]],
        depths: Optional[List[np.ndarray]],
        labels: Optional[List[np.ndarray]],
        masks: Optional[List[np.ndarray]],
    ) -> None:
        """TODO"""
        if camera_poses is not None:
            self._camera_poses.extend(camera_poses)
        if intrinsics is not None:
            self._intrinsics.extend(intrinsics)
        if images is not None:
            self._images.extend(images)
        if depths is not None:
            self._depths.extend(depths)
        if labels is not None:
            self._labels.extend(labels)
        if masks is not None:
            self._masks.extend(masks)

    @abstractmethod
    def postprocess_data(self) -> None:
        """TODO"""
        raise NotImplementedError

    def _create_data_directories(self) -> None:
        for path in self._data_directory_paths:
            if not os.path.exists(path):
                os.mkdir(path)

    @abstractmethod
    def save_data(self) -> None:
        """TODO"""
        raise NotImplementedError
