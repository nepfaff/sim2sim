from typing import Tuple, Union, List
from itertools import chain

import numpy as np
from pydrake.all import (
    MultibodyPlant,
    RigidTransform,
    DepthRenderCamera,
    ClippingRange,
    DepthRange,
    RenderCameraCore,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    CameraInfo,
    RgbdSensor,
    DiagramBuilder,
    SceneGraph,
    Simulator,
)

from .image_generator_base import ImageGeneratorBase
from .cameras import generate_camera_pose_circle


class SphereImageGenerator(ImageGeneratorBase):
    """
    TODO
    Talk about this being cheating (spawn cameras)
    """

    def __init__(
        self,
        builder: DiagramBuilder,
        plant: MultibodyPlant,
        scene_graph: SceneGraph,
        look_at_point: np.ndarray,
        z_distances: np.ndarray,
        radii: np.ndarray,
        num_poses: np.ndarray,
    ):
        """
        :param builder: The diagram builder.
        :param plant: The finalized plant.
        :param scene_graph: The scene graph.
        :param look_at_point: The point that the cameras should look at of shape (3,).
        :param z_distances: The vertical distances (m) of the camera circles from `look_at_point` of shape (n,) where n
            is the number of camera circles. It is recommended to have distances increase monotonically.
        :param radii: The radii (m) of the camera circles of shape (n,) where n is the number of camera circles. It is
            recommended to have radii decrease monotonically.
        :param num_poses: The number of poses for each camera circle of shape (n,) where n is the number of camera
            circles. The number of poses should decrease as the radius decreases.
        """
        super().__init__(builder, plant, scene_graph)

        assert len(z_distances) == len(radii) and len(z_distances) == len(
            num_poses
        ), "'z_distances', 'radii', and 'num_poses' must have the same length."

        self._look_at_point = look_at_point
        self._z_distances = z_distances
        self._radii = radii
        self._num_poses = num_poses

        # TODO: Does an image generator ever need the plant? If not, then remove from this and from base class

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

    def generate_images(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:
        camera_poses = self._generate_camera_poses()

        # TODO: Clean this up!

        # TODO: Move this rendering logic into the base class constructor
        renderer = "my_renderer"
        if not self._scene_graph.HasRenderer(renderer):
            self._scene_graph.AddRenderer(renderer, MakeRenderEngineVtk(RenderEngineVtkParams()))

        # Add cameras to scene
        parent_frame_id = self._scene_graph.world_frame_id()
        camera_info = CameraInfo(width=640, height=480, fov_y=np.pi / 4.0)
        intrinsics = np.array(
            [
                [camera_info.focal_x(), 0.0, camera_info.center_x()],
                [0.0, camera_info.focal_y(), camera_info.center_y()],
                [0.0, 0.0, 1.0],
            ]
        )
        for i, X_WC in enumerate(camera_poses):
            depth_camera = DepthRenderCamera(
                RenderCameraCore(
                    renderer,
                    camera_info,
                    ClippingRange(near=0.1, far=10.0),
                    RigidTransform(),
                ),
                DepthRange(0.1, 10.0),
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

        # TODO: Add light sources everywhere in scene to ensure consistent lighting

        diagram = self._builder.Build()

        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()

        # Take images
        images, depths, masks = [], [], []
        for i in range(len(camera_poses)):
            rgba_image = diagram.GetOutputPort(f"{i}_rgba_image").Eval(context).data
            rgb_image = rgba_image[:, :, :3]
            depth_image = diagram.GetOutputPort(f"{i}_depth_image").Eval(context).data.squeeze()
            label_image = diagram.GetOutputPort(f"{i}_label_image").Eval(context).data.squeeze()

            images.append(rgb_image)
            depths.append(depth_image)
            masks.append(label_image)

        return images, np.broadcast_to(intrinsics, (len(images), 3, 3)), camera_poses, depths, masks
