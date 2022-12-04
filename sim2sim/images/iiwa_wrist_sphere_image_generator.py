from typing import Tuple, List, Union
import copy
import os

import numpy as np
from pydrake.all import (
    RigidTransform,
    CameraInfo,
    DiagramBuilder,
    SceneGraph,
    Simulator,
)

from sim2sim.logging import DynamicLoggerBase
from sim2sim.util import convert_camera_poses_to_iiwa_eef_poses
from .sphere_image_generator import SphereImageGenerator


class IIWAWristSphereImageGenerator(SphereImageGenerator):
    """
    An image generator that uses the iiwa wrist camera for generating camera data. Camera waypoints form a sphere around
    a target point. Also returns ground truth object masks.
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
        super().__init__(
            builder,
            scene_graph,
            logger,
            simulate_time,
            look_at_point,
            z_distances,
            radii,
            num_poses,
        )

        # TODO: Make this an argument
        self._time_between_camera_waypoints = 1.0

        # Create meshcat
        self._visualizer, self._meshcat = self._logger.add_meshcat_visualizer(builder, scene_graph, kProximity=False)

        # We aren't modifying the diagram but are using the existing wrist camera
        self._diagram = builder.Build()

    def _simulate_and_get_image_data(
        self,
        X_WGs: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Moves the robot to generate camera data with the wrist camera."""

        # Get required systems
        plant = self._diagram.GetSubsystemByName("plant")
        world_frame = plant.world_frame()
        gripper_frame = plant.GetFrameByName("body")
        iiwa_trajectory_source = self._diagram.GetSubsystemByName("iiwa_joint_trajectory_source")
        iiwa_trajectory_source.set_meshcat(self._meshcat)

        # Simulate before generating image data
        self._visualizer.StartRecording()
        simulator = Simulator(self._diagram)
        simulator.AdvanceTo(self._simulate_time)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyContextFromRoot(context)

        X_CWs, images, depths, labels, masks = [], [], [], [], []
        X_WG_last = plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=gripper_frame)
        for X_WG in X_WGs:
            iiwa_trajectory_source.set_t_start(context.get_time())
            iiwa_path = [X_WG_last, RigidTransform(X_WG)]
            try:
                iiwa_trajectory_source.set_trajectory(
                    iiwa_path,
                    time_between_breakpoints=self._time_between_camera_waypoints,
                    ik_position_tolerance=0.02,
                    ik_orientation_tolerance=0.02,
                    allow_no_ik_sols=False,
                    debug=True,
                )
            except:
                # Try to skip failed IK solutions
                continue
            X_WG_last = RigidTransform(X_WG)

            simulator.AdvanceTo(context.get_time() + self._time_between_camera_waypoints)

            # Get actual wrist camera pose
            X_WG_actual = plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=gripper_frame)
            X_CWs.append(np.linalg.inv(X_WG_actual.GetAsMatrix4()))

            # Need to make a copy as the original value changes with the simulation
            rgba_image = copy.deepcopy(self._diagram.GetOutputPort("wrist_camera_rgb_image").Eval(context).data)
            rgb_image = rgba_image[:, :, :3]
            images.append(rgb_image)

            depth_image = copy.deepcopy(
                self._diagram.GetOutputPort("wrist_camera_depth_image").Eval(context).data.squeeze()
            )
            depth_image[depth_image == np.inf] = self._max_depth_range
            depths.append(depth_image)

            label_image = copy.deepcopy(
                self._diagram.GetOutputPort("wrist_camera_label_image").Eval(context).data.squeeze()
            )
            object_labels = np.unique(label_image)
            labels.append(object_labels)
            masks.append([np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels])

        self._visualizer.StopRecording()
        self._visualizer.PublishRecording()

        # TODO: Move this to the logger
        html = self._meshcat.StaticHtml()
        with open(os.path.join(self._logger._logging_path, "image_generation.html"), "w") as f:
            f.write(html)

        return (
            X_CWs,
            images,
            depths,
            labels,
            masks,
        )

    def generate_images(
        self,
    ) -> Tuple[
        List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
    ]:

        X_CW = self._generate_camera_poses()
        X_WG = convert_camera_poses_to_iiwa_eef_poses(X_CW)

        # TODO: Refactor this so that script argument specifies these 3 params and both make station and this function
        # uses the same ones
        camera_info = CameraInfo(width=1920, height=1440, fov_y=np.pi / 4.0)
        intrinsics = np.array(
            [
                [camera_info.focal_x(), 0.0, camera_info.center_x()],
                [0.0, camera_info.focal_y(), camera_info.center_y()],
                [0.0, 0.0, 1.0],
            ]
        )

        X_CWs, images, depths, labels, masks = self._simulate_and_get_image_data(X_WG)

        camera_poses_lst = list(X_CWs)
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