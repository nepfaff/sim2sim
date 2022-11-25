from typing import Tuple, List, Optional
import os
import datetime
import yaml

import numpy as np
from pydrake.all import (
    DiagramBuilder,
    SceneGraph,
    Meshcat,
    StartMeshcat,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    ContactVisualizerParams,
    ContactVisualizer,
    MultibodyPlant,
)
from PIL import Image

from sim2sim.logging import DynamicLoggerBase


class DynamicLogger(DynamicLoggerBase):
    def __init__(self, logging_frequency_hz: float, logging_path: str, label_to_mask: int):
        """
        :param logging_frequency_hz: The frequency at which we want to log at.
        :param logging_path: The path to the directory that we want to write the log files to.
        :param label_to_mask: The label that we want to save binary masks for.
        """
        super().__init__(logging_frequency_hz, logging_path)

        self._label_to_mask = label_to_mask

    @staticmethod
    def _add_meshcat_visualizer(
        builder: DiagramBuilder, scene_graph: SceneGraph, kProximity: bool = False
    ) -> Tuple[MeshcatVisualizer, Meshcat]:
        """
        Adds a meshcat visualizer to `builder`.

        :param builder: The diagram builder to add the visualizer to.
        :param scene_graph: The scene graph of the scene to visualize.
        :param kProximity: Whether to visualize kProximity or kIllustration. Visualize kProximity if true.
        :return: A tuple of (visualizer, meshcat).
        """
        meshcat = StartMeshcat()
        meshcat_params = MeshcatVisualizerParams()
        meshcat_params.role = Role.kProximity if kProximity else Role.kIllustration
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph.get_query_output_port(), meshcat, meshcat_params
        )
        return visualizer, meshcat

    @staticmethod
    def _add_contact_visualizer(builder: DiagramBuilder, meshcat: Meshcat, plant: MultibodyPlant) -> None:
        """
        Adds a contact visualizer to `builder`.

        :param builder: The diagram builder to add the visualizer to.
        :param meshcat: The meshcat that we want to add the contact visualizer to.
        :param plant: The plant for which we want to visualize contact forces.
        """
        cparams = ContactVisualizerParams()
        cparams.force_threshold = 1e-2
        cparams.newtons_per_meter = 1e6
        cparams.newton_meters_per_meter = 1e1
        cparams.radius = 0.002
        _ = ContactVisualizer.AddToBuilder(builder, plant, meshcat, cparams)

    def add_visualizers(
        self, builder: DiagramBuilder, scene_graph: SceneGraph, is_outer: bool
    ) -> Tuple[MeshcatVisualizer, Meshcat]:
        """
        Add visualizers.
        :param builder: The diagram builder to add the visualizer to.
        :param scene_graph: The scene graph of the scene to visualize.
        :return: A tuple of (visualizer, meshcat).
        """
        visualizer, meshcat = self._add_meshcat_visualizer(builder, scene_graph)
        if self._inner_plant is not None and self._outer_plant is not None:
            self._add_contact_visualizer(builder, meshcat, self._outer_plant if is_outer else self._inner_plant)
        return visualizer, meshcat

    def log(
        self,
        camera_poses: Optional[List[np.ndarray]],
        intrinsics: Optional[List[np.ndarray]],
        images: Optional[List[np.ndarray]],
        depths: Optional[List[np.ndarray]],
        labels: Optional[List[np.ndarray]],
        masks: Optional[List[np.ndarray]],
    ) -> None:
        super().log(
            camera_poses=camera_poses, intrinsics=intrinsics, images=images, depths=depths, labels=labels, masks=masks
        )

    def postprocess_data(self) -> None:
        raise NotImplementedError

    def save_data(self) -> None:
        self._create_data_directories()

        meta_data = {
            "creation_timestamp": self._creation_timestamp,
            "logging_timestamp": str(datetime.datetime.now()),
        }

        lengths = [
            len(self._camera_poses),
            len(self._intrinsics),
            len(self._images),
            len(self._depths),
            len(self._labels),
            len(self._masks),
        ]
        assert all(l == lengths[0] for l in lengths), f"All camera data must have the same length. Lengths: {lengths}"

        for i, (pose, intrinsics, image, depth, labels, masks) in enumerate(
            zip(self._camera_poses, self._intrinsics, self._images, self._depths, self._labels, self._masks)
        ):

            np.savetxt(os.path.join(self._camera_poses_dir_path, f"pose{i:04d}.txt"), pose)
            np.savetxt(os.path.join(self._intrinsics_dir_path, f"intrinsics{i:04d}.txt"), intrinsics)
            np.savetxt(os.path.join(self._depths_dir_path, f"depth{i:04d}.txt"), depth)

            image_pil = Image.fromarray(image)
            image_pil.save(os.path.join(self._images_dir_path, f"image{i:04d}.png"))

            for label, mask in zip(labels, masks):
                if label == self._label_to_mask:
                    mask_pil = Image.fromarray(mask)
                    mask_pil.save(os.path.join(self._masks_dir_path, f"mask{i:04d}.png"))

        with open(self._meta_data_file_path, "w") as f:
            yaml.dump(meta_data, f)
