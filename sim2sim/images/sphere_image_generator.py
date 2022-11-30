from typing import Tuple, List, Union
import copy

import numpy as np
from pydrake.all import (
    RigidTransform,
    DepthRenderCamera,
    ClippingRange,
    DepthRange,
    RenderCameraCore,
    CameraInfo,
    RgbdSensor,
    DiagramBuilder,
    SceneGraph,
    Simulator,
)

from sim2sim.logging import DynamicLoggerBase
from .image_generator_base import ImageGeneratorBase
from .cameras import generate_camera_pose_circle


class SphereImageGenerator(ImageGeneratorBase):
    """
    An image generator that spawns cameras in a sphere around a target point for generating camera data. Also returns
    ground truth object masks.
    """

    def __init__(
        self,
        builder: DiagramBuilder,
        scene_graph: SceneGraph,
        logger: DynamicLoggerBase,
        simulate_time: float,
        look_at_point: Union[List, np.ndarray],
        z_distances: Union[List, np.ndarray],
        radii: Union[List, np.ndarray],
        num_poses: Union[List, np.ndarray],
    ):
        """
        :param builder: The diagram builder.
        :param scene_graph: The scene graph.
        :param logger: The logger.
        :param simulate_time: The time in seconds to simulate before generating the image data.
        :param look_at_point: The point that the cameras should look at of shape (3,).
        :param z_distances: The vertical distances (m) of the camera circles from `look_at_point` of shape (n,) where n
            is the number of camera circles. It is recommended to have distances increase monotonically.
        :param radii: The radii (m) of the camera circles of shape (n,) where n is the number of camera circles. It is
            recommended to have radii decrease monotonically.
        :param num_poses: The number of poses for each camera circle of shape (n,) where n is the number of camera
            circles. The number of poses should decrease as the radius decreases.
        """
        super().__init__(builder, scene_graph, logger)

        assert len(z_distances) == len(radii) and len(z_distances) == len(
            num_poses
        ), "'z_distances', 'radii', and 'num_poses' must have the same length."

        self._simulate_time = simulate_time
        self._look_at_point = look_at_point if isinstance(look_at_point, np.ndarray) else np.asarray(look_at_point)
        self._z_distances = z_distances if isinstance(z_distances, np.ndarray) else np.asarray(z_distances)
        self._radii = radii if isinstance(radii, np.ndarray) else np.asarray(radii)
        self._num_poses = num_poses if isinstance(num_poses, np.ndarray) else np.asarray(num_poses)

        self._min_depth_range = 0.1
        self._max_depth_range = 10.0

    def _generate_camera_poses(self) -> np.ndarray:
        """
        :return: Homogenous cam2world transforms of shape (n,4,4) where n is the number of camera poses.
        """
        camera_poses = []
        for z_dist, radius, num_poses in zip(self._z_distances, self._radii, self._num_poses):
            X_WCs = generate_camera_pose_circle(
                look_at_point=self._look_at_point,
                camera_location_center=self._look_at_point + [0.0, 0.0, z_dist],
                radius=radius,
                num_cam_poses=num_poses,
            )
            camera_poses.append(X_WCs)
        return np.concatenate(camera_poses, axis=0)

    def _add_cameras(self, camera_poses: np.ndarray, camera_info: CameraInfo) -> None:
        """Adds depth cameras to the scene."""
        parent_frame_id = self._scene_graph.world_frame_id()
        for i, X_WC in enumerate(camera_poses):
            depth_camera = DepthRenderCamera(
                RenderCameraCore(
                    self._renderer,
                    camera_info,
                    ClippingRange(near=self._min_depth_range, far=self._max_depth_range),
                    RigidTransform(),
                ),
                DepthRange(self._min_depth_range, self._max_depth_range),
            )
            rgbd = self._builder.AddSystem(
                RgbdSensor(
                    parent_id=parent_frame_id,
                    X_PB=RigidTransform(X_WC),
                    depth_camera=depth_camera,
                    show_window=False,
                )
            )
            self._builder.Connect(self._scene_graph.get_query_output_port(), rgbd.query_object_input_port())

            # Export the camera outputs
            self._builder.ExportOutput(rgbd.color_image_output_port(), f"{i}_rgba_image")
            self._builder.ExportOutput(rgbd.depth_image_32F_output_port(), f"{i}_depth_image")
            self._builder.ExportOutput(rgbd.label_image_output_port(), f"{i}_label_image")

    def _simulate_and_get_image_data(
        self,
        camera_poses: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Simulates before taking camera data."""
        assert self._diagram is not None, "Must build the diagram before generating image data."

        # Simulate before generating image data
        simulator = Simulator(self._diagram)
        simulator.AdvanceTo(self._simulate_time)
        context = simulator.get_mutable_context()

        images, depths, labels, masks = [], [], [], []
        for i in range(len(camera_poses)):
            # Need to make a copy as the original value changes with the simulation
            rgba_image = copy.deepcopy(self._diagram.GetOutputPort(f"{i}_rgba_image").Eval(context).data)
            rgb_image = rgba_image[:, :, :3]
            images.append(rgb_image)

            depth_image = copy.deepcopy(self._diagram.GetOutputPort(f"{i}_depth_image").Eval(context).data.squeeze())
            depth_image[depth_image == np.inf] = self._max_depth_range
            depths.append(depth_image)

            label_image = copy.deepcopy(self._diagram.GetOutputPort(f"{i}_label_image").Eval(context).data.squeeze())
            object_labels = np.unique(label_image)
            labels.append(object_labels)
            masks.append([np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels])

        return (
            images,
            depths,
            labels,
            masks,
        )

    def generate_images(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        camera_poses = self._generate_camera_poses()
        camera_info = CameraInfo(width=640, height=480, fov_y=np.pi / 4.0)
        intrinsics = np.array(
            [
                [camera_info.focal_x(), 0.0, camera_info.center_x()],
                [0.0, camera_info.focal_y(), camera_info.center_y()],
                [0.0, 0.0, 1.0],
            ]
        )
        self._add_cameras(camera_poses, camera_info)

        self._diagram = self._builder.Build()

        images, depths, labels, masks = self._simulate_and_get_image_data(camera_poses)

        camera_poses_lst = list(camera_poses)
        intrinsics_broadcasted = list(np.broadcast_to(intrinsics, (len(images), 3, 3)))
        self._logger.log(
            camera_poses=camera_poses_lst,
            intrinsics=intrinsics_broadcasted,
            images=images,
            depths=depths,
            labels=labels,
            masks=masks,
        )

        return (
            images,
            intrinsics_broadcasted,
            camera_poses_lst,
            depths,
            labels,
            masks,
        )