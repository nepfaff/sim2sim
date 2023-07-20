from typing import Tuple, List, Union, Optional
import os
from collections import OrderedDict

import numpy as np
from pydrake.all import (
    RigidTransform,
    CameraInfo,
    DiagramBuilder,
    SceneGraph,
    Simulator,
    LoadIrisRegionsYamlFile,
    HPolyhedron,
    MathematicalProgram,
    Solve,
    IrisInConfigurationSpace,
    IrisOptions,
    RollPitchYaw,
    MultibodyPlant,
    Context,
    GraphOfConvexSetsOptions,
    GcsTrajectoryOptimization,
    Point,
    PiecewisePolynomial,
)
from manipulation.meshcat_utils import AddMeshcatTriad

from sim2sim.logging import DynamicLogger
from sim2sim.util import (
    convert_camera_poses_to_iiwa_eef_poses,
    compute_joint_angles_for_eef_poses,
    IIWAJointTrajectorySource,
    calc_inverse_kinematics,
)
from sim2sim.images import SphereImageGenerator


def scale_hpolyhedron(hpoly, scale_factor):
    # Shift to the center.
    xc = hpoly.ChebyshevCenter()
    A = hpoly.A()
    b = hpoly.b() - A @ xc
    # Scale
    b = scale_factor * b
    # Shift back
    b = b + A @ xc
    return HPolyhedron(A, b)


def check_non_empty(region):
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(region.ambient_dimension())
    region.AddPointInSetConstraints(prog, x)
    result = Solve(prog)
    assert result.is_success()


# TODO: Clean up + move to utils
def compute_iris_region(
    plant: MultibodyPlant,
    plant_context: Context,
    iris_options: IrisOptions,
    existing_regions: dict,
    name: str,
    seed: np.ndarray,
    use_existing_regions_as_obstacles: bool = True,
    regions_as_obstacles_scale_factor: float = 0.95,
):
    plant.SetPositions(
        context=plant_context,
        model_instance=plant.GetModelInstanceByName("iiwa"),
        q=seed,
    )
    if use_existing_regions_as_obstacles:
        iris_options.configuration_obstacles = [
            scale_hpolyhedron(r, regions_as_obstacles_scale_factor)
            for k, r in existing_regions.items()
            if k != name
        ]
        for h in iris_options.configuration_obstacles:
            check_non_empty(h)
    else:
        iris_options.configuration_obstacles = None
    hpoly = IrisInConfigurationSpace(plant, plant_context, iris_options)

    check_non_empty(hpoly)
    reduced = hpoly.ReduceInequalities()
    check_non_empty(reduced)

    return reduced


# TODO: Clean up + move this to utils
def generate_iris_regions(
    plant: MultibodyPlant,
    plant_context: Context,
    iiwa_controller_plant: MultibodyPlant,
    iris_options: IrisOptions,
) -> dict:
    seeds = OrderedDict()
    seeds["Nominal position"] = np.array([1.5, -0.4, 0.0, -1.75, 0.0, 1.5, 0.0])
    seeds["Swan position"] = calc_inverse_kinematics(
        plant=iiwa_controller_plant,
        X_G=RigidTransform(RollPitchYaw(-2.05, 0.01, 3.09), [-0.04, -0.01, 0.67]),
        initial_guess=seeds["Nominal position"],
        position_tolerance=0.05,
        orientation_tolerance=0.05,
    )
    seeds["Above (0,0.5)"] = calc_inverse_kinematics(
        plant=iiwa_controller_plant,
        X_G=RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [0.0, 0.5, 0.4]),
        initial_guess=seeds["Nominal position"],
        position_tolerance=0.05,
        orientation_tolerance=0.05,
    )
    seeds["Above (0.25,0.25)"] = calc_inverse_kinematics(
        plant=iiwa_controller_plant,
        X_G=RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [0.25, 0.25, 0.4]),
        initial_guess=seeds["Above (0,0.5)"],
        position_tolerance=0.05,
        orientation_tolerance=0.05,
    )
    # seeds["Above (0.5,0), Above container"] = calc_inverse_kinematics(
    #     plant=iiwa_controller_plant,
    #     X_G=RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [0.5, 0.0, 0.4]),
    #     initial_guess=seeds["Above (0.25,0.25)"],
    #     position_tolerance=0.05,
    #     orientation_tolerance=0.05,
    # )
    seeds["Above container, sliding start"] = calc_inverse_kinematics(
        plant=iiwa_controller_plant,
        X_G=RigidTransform(RollPitchYaw(-1.90, 0.01, 3.10), [0.5, 0.29, 0.38]),
        initial_guess=seeds["Above (0.25,0.25)"],
        position_tolerance=0.05,
        orientation_tolerance=0.05,
    )
    seeds["Above (-0.25,0.25)"] = calc_inverse_kinematics(
        plant=iiwa_controller_plant,
        X_G=RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [-0.25, 0.25, 0.4]),
        initial_guess=seeds["Above (0,0.5)"],
        position_tolerance=0.05,
        orientation_tolerance=0.05,
    )
    # seeds["Way to bin"] = calc_inverse_kinematics(
    #     plant=iiwa_controller_plant,
    #     X_G=RigidTransform(RollPitchYaw(-1.83, 0.01, 3.10), [-0.56, 0.23, 0.38]),
    #     initial_guess=seeds["Above (-0.25,0.25)"],
    #     position_tolerance=0.05,
    #     orientation_tolerance=0.05,
    # )
    # seeds["Above (-0.5,0), Above bin"] = calc_inverse_kinematics(
    #     plant=iiwa_controller_plant,
    #     X_G=RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [-0.5, 0.0, 0.4]),
    #     initial_guess=seeds["Above (-0.25,0.25)"],
    #     position_tolerance=0.05,
    #     orientation_tolerance=0.05,
    # )

    iris_regions = dict()
    for name, seed in seeds.items():
        print(f"Computing IRIS region '{name}'")
        iris_regions[name] = compute_iris_region(
            plant=plant,
            plant_context=plant_context,
            iris_options=iris_options,
            existing_regions=iris_regions,
            name=name,
            seed=seed,
        )
    return iris_regions


def solve_gcs_trajectory_optimization(
    plant: MultibodyPlant, iris_regions: dict, q_start: np.ndarray, q_goal: np.ndarray
) -> Union[PiecewisePolynomial, None]:
    assert len(q_start) == len(q_goal)
    assert len(q_start) == iris_regions[next(iter(iris_regions))].ambient_dimension()

    gcs = GcsTrajectoryOptimization(len(q_start))
    regions = gcs.AddRegions(list(iris_regions.values()), order=1)
    source = gcs.AddRegions([Point(q_start)], order=0)
    target = gcs.AddRegions([Point(q_goal)], order=0)
    gcs.AddEdges(source, regions)
    gcs.AddEdges(regions, target)
    gcs.AddTimeCost()
    gcs.AddVelocityBounds(
        np.clip(plant.GetVelocityLowerLimits(), a_min=-10, a_max=10),
        np.clip(plant.GetVelocityUpperLimits(), a_min=-10, a_max=10),
    )

    options = GraphOfConvexSetsOptions()
    options.preprocessing = True
    options.convex_relaxation = True
    options.max_rounded_paths = 15
    traj, result = gcs.SolvePath(source, target, options)
    return traj if result.is_success() else None


class IIWAWristSphereImageGenerator(SphereImageGenerator):
    """
    An image generator that uses the iiwa wrist camera for generating camera data.
    Camera waypoints form a sphere around a target point. Also returns ground truth
    object masks.
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
        use_gcs: bool,
        iris_regions_path: Optional[str] = None,
    ):
        """
        :param builder: The diagram builder.
        :param scene_graph: The scene graph.
        :param logger: The logger.
        :param simulate_time: The time in seconds to simulate before generating the
            image data.
        :param look_at_point: The point that the cameras should look at of shape (3,).
        :param z_distances: The vertical distances (m) of the camera circles from
            `look_at_point` of shape (n,) where n is the number of camera circles. It is
            recommended to have distances increase monotonically.
        :param radii: The radii (m) of the camera circles of shape (n,) where n is the
            number of camera circles. It is recommended to have radii decrease
            monotonically.
        :param num_poses: The number of poses for each camera circle of shape (n,) where
            n is the number of camera circles. The number of poses should decrease as
            the radius decreases.
        :param time_between_camera_waypoints: The time in seconds that the iiwa should
            take to move from one wrist camera waypoint to the next.
        :param has_leg_camera: Whether the setup has an iiwa leg camera of name
            `camera_leg`.
        :param num_cameras_below_table: The number of cameras below the table.
            NOTE: The table must have no visual element for these cameras to produce
            useful data. These cameras must have name `camera_below_table_{i}` where i
            is an index in range 0...num_cameras_below_table-1.
        :param use_gcs: Whether to use GCS for motion planing. Otherwise, simple
            interpolation without collision awareness is used.
        :param iris_regions_path: The path to the yaml file containing already computed
            IRIS regions to use for GCS motion planing. This argument is ignored if
            `use_gcs` is false. If None, the regions are computed from scratch.
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
        self._use_gcs = use_gcs

        # Create meshcat
        self._visualizer, self._meshcat = self._logger.add_meshcat_visualizer(
            builder, scene_graph, kProximity=False
        )

        # We aren't modifying the diagram but are using the existing wrist camera
        self._diagram = builder.Build()

        # Ensure that have IRIS regions when using GCS
        if use_gcs:
            if iris_regions_path is None:
                simulator = Simulator(self._diagram)
                simulator.AdvanceTo(self._simulate_time)
                plant = self._diagram.GetSubsystemByName("plant")
                context = simulator.get_mutable_context()
                plant_context = plant.GetMyContextFromRoot(context)
                iiwa_controller_plant = self._diagram.GetSubsystemByName(
                    "iiwa_inverse_dynamics_controller"
                ).get_multibody_plant_for_control()
                iris_options = IrisOptions()
                iris_options.iteration_limit = 10
                # increase num_collision_infeasible_samples to improve the (probabilistic)
                # certificate of having no collisions.
                iris_options.num_collision_infeasible_samples = 3
                iris_options.require_sample_point_is_contained = True
                iris_options.relative_termination_threshold = 0.01
                iris_options.termination_threshold = -1
                self._iris_regions = generate_iris_regions(
                    plant, plant_context, iiwa_controller_plant, iris_options
                )
            else:
                self._iris_regions = LoadIrisRegionsYamlFile(iris_regions_path)
            logger.log(iris_regions=self._iris_regions)

            # TODO: temp only
            from pydrake.all import SaveIrisRegionsYamlFile

            SaveIrisRegionsYamlFile(
                os.path.join(self._logger._logging_path, "iris_regions.yaml"),
                self._iris_regions,
            )

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
        X_WG_feasible, q_waypoints = compute_joint_angles_for_eef_poses(
            X_WGs,
            iiwa_controller_plant,
            initial_guess=iiwa_trajectory_source._q_nominal,
            ik_position_tolerance=0.02,
            ik_orientation_tolerance=0.01,
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
        if self._use_gcs:
            # TODO: Clean up hardcoded nominal position
            nominal_position = np.array([1.5, -0.4, 0.0, -1.75, 0.0, 1.5, 0.0])
            last_q = calc_inverse_kinematics(
                plant=iiwa_controller_plant,
                X_G=X_WG_last,
                initial_guess=nominal_position,
                position_tolerance=0.05,
                orientation_tolerance=0.05,
            )
            for i, next_q in enumerate(q_waypoints):
                # Concatenate the wsg positions to the iiwa joint positions
                q_traj = solve_gcs_trajectory_optimization(
                    plant=plant,
                    iris_regions=self._iris_regions,
                    q_start=np.concatenate([last_q, [0.0, 0.0]]),
                    q_goal=np.concatenate([next_q, [0.0, 0.0]]),
                )
                if q_traj is not None:
                    iiwa_trajectory_source.set_trajectory(
                        q_traj, context.get_time() + q_traj.start_time()
                    )
                    traj_duration = (
                        q_traj.end_time()
                        - q_traj.start_time()
                        + self._time_between_camera_waypoints
                    )
                    simulator.AdvanceTo(context.get_time() + traj_duration)
                    last_q = next_q
                else:
                    # Try to generate a new IRIS region

                    ### TMP start
                    simulator1 = Simulator(self._diagram)
                    simulator1.AdvanceTo(self._simulate_time)
                    plant1 = self._diagram.GetSubsystemByName("plant")
                    context1 = simulator1.get_mutable_context()
                    plant_context1 = plant1.GetMyContextFromRoot(context1)
                    iris_options = IrisOptions()
                    iris_options.iteration_limit = 10
                    # increase num_collision_infeasible_samples to improve the (probabilistic)
                    # certificate of having no collisions.
                    iris_options.num_collision_infeasible_samples = 3
                    iris_options.require_sample_point_is_contained = True
                    iris_options.relative_termination_threshold = 0.01
                    iris_options.termination_threshold = -1
                    name = f"camera_waypoint_{i}"
                    print(f"Computing IRIS region '{name}'")
                    try:
                        self._iris_regions[name] = compute_iris_region(
                            plant=plant1,
                            plant_context=plant_context1,
                            iris_options=iris_options,
                            existing_regions=self._iris_regions,
                            name=name,
                            seed=next_q,
                        )
                        from pydrake.all import SaveIrisRegionsYamlFile

                        SaveIrisRegionsYamlFile(
                            "./my_iris.yaml",
                            self._iris_regions,
                        )
                    except:
                        # Seed point is alrady inside a region
                        pass
                    ### TMP end

                    num_skipped += 1
                    continue
        else:
            # TODO: Refactor to avoid code duplication with GCS variant
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
