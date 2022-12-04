from typing import Tuple, List, Optional
import os
import datetime
import yaml

import numpy as np
import open3d as o3d
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
    LogVectorOutput,
    Context,
)
from PIL import Image

from sim2sim.logging import DynamicLoggerBase


class DynamicLogger(DynamicLoggerBase):
    def __init__(
        self, logging_frequency_hz: float, logging_path: str, kProximity: bool, label_to_mask: int, manipuland_name: str
    ):
        """
        :param logging_frequency_hz: The frequency at which we want to log at.
        :param logging_path: The path to the directory that we want to write the log files to.
        :param kProximity: Whether to visualize kProximity or kIllustration. Visualize kProximity if true.
        :param label_to_mask: The label that we want to save binary masks for.
        :param manipuland_name: The name of the manipuland. Required for pose logging.
        """
        super().__init__(logging_frequency_hz, logging_path, kProximity)

        self._label_to_mask = label_to_mask
        self._manipuland_name = manipuland_name

        # Pose data logs
        self._outer_manipuland_pose_logger = None
        self._outer_manipuland_poses: np.ndarray = None
        self._outer_manipuland_pose_times: np.ndarray = None
        self._inner_manipuland_pose_logger = None
        self._inner_manipuland_poses: np.ndarray = None
        self._inner_manipuland_pose_times: np.ndarray = None

    @staticmethod
    def add_meshcat_visualizer(
        builder: DiagramBuilder, scene_graph: SceneGraph, kProximity: bool
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
        visualizer, meshcat = self.add_meshcat_visualizer(builder, scene_graph, self._kProximity)
        if self._inner_plant is not None and self._outer_plant is not None:
            self._add_contact_visualizer(builder, meshcat, self._outer_plant if is_outer else self._inner_plant)
        return visualizer, meshcat

    def add_pose_logging(self, outer_builder: DiagramBuilder, inner_builder: DiagramBuilder) -> None:
        self._outer_manipuland_pose_logger = LogVectorOutput(
            self._outer_plant.get_state_output_port(self._outer_plant.GetModelInstanceByName(self._manipuland_name)),
            outer_builder,
            1 / self._logging_frequency_hz,
        )
        self._inner_manipuland_pose_logger = LogVectorOutput(
            self._inner_plant.get_state_output_port(self._inner_plant.GetModelInstanceByName(self._manipuland_name)),
            inner_builder,
            1 / self._logging_frequency_hz,
        )

    def log_poses(self, context: Context, is_outer: bool) -> None:
        assert self._outer_manipuland_pose_logger is not None and self._inner_manipuland_pose_logger is not None

        if is_outer:
            log = self._outer_manipuland_pose_logger.FindLog(context)
            self._outer_manipuland_pose_times = log.sample_times()
            self._outer_manipuland_poses = log.data().T  # Shape (t, 13)
        else:
            log = self._inner_manipuland_pose_logger.FindLog(context)
            self._inner_manipuland_pose_times = log.sample_times()
            self._inner_manipuland_poses = log.data().T  # Shape (t, 13)

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
        super().log(
            camera_poses=camera_poses,
            intrinsics=intrinsics,
            images=images,
            depths=depths,
            labels=labels,
            masks=masks,
            raw_mesh=raw_mesh,
            processed_mesh=processed_mesh,
        )

    def postprocess_data(self) -> None:
        raise NotImplementedError

    def save_mesh_data(self) -> Tuple[str, str]:
        """
        Saves the raw and processed meshes if they exist.

        :return: A tuple of (raw_mesh_file_path, processed_mesh_file_path).
        """
        raw_mesh_file_path = os.path.join(self._mesh_dir_path, "raw_mesh.obj")
        processed_mesh_file_path = os.path.join(self._mesh_dir_path, "processed_mesh.obj")
        if self._raw_mesh:
            o3d.io.write_triangle_mesh(raw_mesh_file_path, self._raw_mesh)
        if self._processed_mesh:
            o3d.io.write_triangle_mesh(processed_mesh_file_path, self._processed_mesh)
        return raw_mesh_file_path, processed_mesh_file_path

    def save_data(self) -> None:
        # Meta data
        meta_data = {
            "logger_creation_timestamp": self._creation_timestamp,
            "logging_timestamp": str(datetime.datetime.now()),
        }
        with open(self._meta_data_file_path, "w") as f:
            yaml.dump(meta_data, f)

        # Camera data
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

            mask_pil = None
            for label, mask in zip(labels, masks):
                if label == self._label_to_mask:
                    mask_pil = Image.fromarray(mask)
            if mask_pil is None:
                # Save black image
                mask_pil = Image.new("RGB", (image_pil.width, image_pil.height))
            mask_pil.save(os.path.join(self._masks_dir_path, f"mask{i:04d}.png"))

        np.savetxt(os.path.join(self._logging_path, "outer_manipuland_poses.txt"), self._outer_manipuland_poses)
        np.savetxt(
            os.path.join(self._logging_path, "outer_manipuland_pose_times.txt"), self._outer_manipuland_pose_times
        )
        np.savetxt(os.path.join(self._logging_path, "inner_manipuland_poses.txt"), self._inner_manipuland_poses)
        np.savetxt(
            os.path.join(self._logging_path, "inner_manipuland_pose_times.txt"), self._inner_manipuland_pose_times
        )

        # Mesh data
        self.save_mesh_data()
