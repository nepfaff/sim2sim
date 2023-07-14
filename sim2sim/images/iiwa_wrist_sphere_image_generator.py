from typing import Tuple, List, Union
import os

import numpy as np
from pydrake.all import (
    RigidTransform,
    CameraInfo,
    DiagramBuilder,
    SceneGraph,
    Simulator,
)
from manipulation.meshcat_utils import AddMeshcatTriad

from sim2sim.logging import DynamicLogger
from sim2sim.util import (
    convert_camera_poses_to_iiwa_eef_poses,
    prune_infeasible_eef_poses,
    IIWAJointTrajectorySource,
)
from sim2sim.images import SphereImageGenerator


class IIWAWristSphereImageGenerator(SphereImageGenerator):
    """
    An image generator that uses the iiwa wrist camera for generating camera data. Camera waypoints form a sphere around
    a target point. Also returns ground truth object masks.
    """

    def __init__(
        self,
        builder: DiagramBuilder,
        scene_graph: SceneGraph,
        logger: DynamicLogger,
        simulate_time: float,
        look_at_point: Union[List, np.ndarray],
        z_distances: Union[List, np.ndarray],
        radii: Union[List, np.ndarray],
        num_poses: Union[List, np.ndarray],
        time_between_camera_waypoints: float,
        has_leg_camera: bool,
        num_cameras_below_table: int,
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
        :param time_between_camera_waypoints: The time in seconds that the iiwa should take to move from one wrist camera
            waypoint to the next.
        :param has_leg_camera: Whether the setup has an iiwa leg camera of name `camera_leg`.
        :param num_cameras_below_table: The number of cameras below the table. NOTE: The table must have no visual element
            for these cameras to produce useful data. These cameras must have name `camera_below_table_{i}` where i is an
            index in range 0...num_cameras_below_table-1.
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

        self._time_between_camera_waypoints = time_between_camera_waypoints
        self._has_leg_camera = has_leg_camera
        self._num_cameras_below_table = num_cameras_below_table

        # Create meshcat
        self._visualizer, self._meshcat = self._logger.add_meshcat_visualizer(
            builder, scene_graph, kProximity=False
        )

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
        iiwa_controller_plant = self._diagram.GetSubsystemByName(
            "iiwa_inverse_dynamics_controller"
        ).get_multibody_plant_for_control()
        iiwa_trajectory_source: IIWAJointTrajectorySource = (
            self._diagram.GetSubsystemByName("iiwa_joint_trajectory_source")
        )
        iiwa_trajectory_source.set_meshcat(self._meshcat)

        # Simulate before generating image data
        self._visualizer.StartRecording()
        simulator = Simulator(self._diagram)
        simulator.AdvanceTo(self._simulate_time)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyContextFromRoot(context)

        X_CWs, images, depths, labels, masks = [], [], [], [], []

        # Get image data from leg camera
        X_WG_actual = plant.CalcRelativeTransform(
            plant_context,
            frame_A=world_frame,
            frame_B=plant.GetFrameByName("camera_leg"),
        )
        X_CWs.append(np.linalg.inv(X_WG_actual.GetAsMatrix4()))
        image, depth_image, object_labels, object_masks = self._get_camera_data(
            "camera_leg", context
        )
        images.append(image)
        depths.append(depth_image)
        labels.append(object_labels)
        masks.append(object_masks)

        # Get image data from below table cameras
        for i in range(self._num_cameras_below_table):
            X_WG_actual = plant.CalcRelativeTransform(
                plant_context,
                frame_A=world_frame,
                frame_B=plant.GetFrameByName(f"camera_below_table_{i}"),
            )
            X_CWs.append(np.linalg.inv(X_WG_actual.GetAsMatrix4()))
            image, depth_image, object_labels, object_masks = self._get_camera_data(
                f"camera_below_table_{i}", context
            )
            images.append(image)
            depths.append(depth_image)
            labels.append(object_labels)
            masks.append(object_masks)

        # Prune camera poses that are not reachable with the wrist camera
        X_WG_feasible = prune_infeasible_eef_poses(
            X_WGs,
            iiwa_controller_plant,
            initial_guess=iiwa_trajectory_source._q_nominal,
            ik_position_tolerance=0.02,
            ik_orientation_tolerance=0.02,
        )
        num_poses = len(X_WGs)
        num_poses_feasible = len(X_WG_feasible)
        print(
            f"Pruned {num_poses-num_poses_feasible} infeasible wrist camera poses. "
            + f"{num_poses_feasible}/{num_poses} poses remaining."
        )

        # Publish the planned camera poses to meshcat
        for i, X_WG in enumerate(X_WG_feasible):
            AddMeshcatTriad(
                self._meshcat, f"X_WG{i:03d}", length=0.15, radius=0.006, X_PT=X_WG
            )

        # Use wrist camera to generate image data
        gripper_frame = plant.GetFrameByName("body")
        X_WG_last = plant.CalcRelativeTransform(
            plant_context, frame_A=world_frame, frame_B=gripper_frame
        )
        num_skipped = 0
        for X_WG in X_WG_feasible:
            iiwa_trajectory_source.set_t_start(context.get_time())
            iiwa_path = [X_WG_last, RigidTransform(X_WG)]
            try:
                iiwa_trajectory_source.compute_and_set_trajectory(
                    iiwa_path,
                    time_between_breakpoints=self._time_between_camera_waypoints,
                    ik_position_tolerance=0.02,
                    ik_orientation_tolerance=0.02,
                    allow_no_ik_sols=False,
                )
            except:
                # Try to skip failed IK solutions
                num_skipped += 1
                continue
            X_WG_last = RigidTransform(X_WG)

            simulator.AdvanceTo(
                context.get_time() + self._time_between_camera_waypoints
            )

            # Get actual wrist camera pose
            X_WG_actual = plant.CalcRelativeTransform(
                plant_context,
                frame_A=world_frame,
                frame_B=plant.GetFrameByName("camera_wrist"),
            )
            X_CWs.append(np.linalg.inv(X_WG_actual.GetAsMatrix4()))

            image, depth_image, object_labels, object_masks = self._get_camera_data(
                "camera_wrist", context
            )
            images.append(image)
            depths.append(depth_image)
            labels.append(object_labels)
            masks.append(object_masks)
        print(
            f"Skipped {num_skipped} of the initially feasible wrist camera poses. "
            + f"{num_poses_feasible-num_skipped}/{num_poses} poses remaining."
        )

        self._visualizer.StopRecording()
        self._visualizer.PublishRecording()

        # TODO: Move this to the logger
        html = self._meshcat.StaticHtml()
        with open(
            os.path.join(self._logger._logging_path, "image_generation.html"), "w"
        ) as f:
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
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        X_CW = self._generate_camera_poses()
        X_WG = convert_camera_poses_to_iiwa_eef_poses(X_CW)

        # TODO: Refactor this so that script argument specifies these 3 params and both
        # make station and this function uses the same ones
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
