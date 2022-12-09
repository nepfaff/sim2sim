from abc import ABC, abstractmethod
from typing import Union, Optional, List
import os
import datetime

import numpy as np
import open3d as o3d
from pydrake.all import MultibodyPlant


class DynamicLoggerBase(ABC):
    """Dynamics logger base class."""

    def __init__(self, logging_frequency_hz: float, logging_path: str, kProximity: bool):
        """
        :param logging_frequency_hz: The frequency at which we want to log at.
        :param logging_path: The path to the directory that we want to write the log files to.
        :param kProximity: Whether to visualize kProximity or kIllustration. Visualize kProximity if true.
        """
        self._logging_frequency_hz = logging_frequency_hz
        self._logging_path = logging_path
        self._kProximity = kProximity

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
        self._mesh_dir_path = os.path.join(logging_path, "meshes")
        self._time_logs_dir_path = os.path.join(logging_path, "time_logs")
        self._data_directory_paths = [
            self._camera_poses_dir_path,
            self._intrinsics_dir_path,
            self._images_dir_path,
            self._depths_dir_path,
            self._masks_dir_path,
            self._mesh_dir_path,
            self._time_logs_dir_path,
        ]
        self._create_data_directories()
        self._meta_data_file_path = os.path.join(logging_path, "meta_data.yaml")

        # Logging data
        self._camera_poses: List[np.ndarray] = []
        self._intrinsics: List[np.ndarray] = []
        self._images: List[np.ndarray] = []
        self._depths: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._masks: List[np.ndarray] = []
        self._raw_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self._processed_mesh: Optional[o3d.geometry.TriangleMesh] = None

    def add_plants(self, outer_plant: MultibodyPlant, inner_plant: MultibodyPlant) -> None:
        """Add finalized plants."""
        self._outer_plant = outer_plant
        self._inner_plant = inner_plant

    @abstractmethod
    def log(
        self,
        camera_poses: Optional[List[np.ndarray]] = None,
        intrinsics: Optional[List[np.ndarray]] = None,
        images: Optional[List[np.ndarray]] = None,
        depths: Optional[List[np.ndarray]] = None,
        labels: Optional[List[np.ndarray]] = None,
        masks: Optional[List[np.ndarray]] = None,
        raw_mesh: Optional[o3d.geometry.TriangleMesh] = None,
        processed_mesh: Optional[o3d.geometry.TriangleMesh] = None,
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
        if raw_mesh is not None:
            self._raw_mesh = raw_mesh
        if processed_mesh is not None:
            self._processed_mesh = processed_mesh

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
