from typing import List, Union, Optional
from enum import Enum

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.all import (
    MultibodyPlant,
    PiecewisePolynomial,
    RigidTransform,
    Solve,
    Meshcat,
    DiagramBuilder,
    MultibodyPlant,
    ModelInstanceIndex,
    PassThrough,
    Demultiplexer,
    InverseDynamicsController,
    Adder,
    StateInterpolatorWithDiscreteDerivative,
    SchunkWsgPositionController,
    MakeMultibodyStateToWsgStateSystem,
    RigidTransform,
    BasicVector,
    LeafSystem,
    RotationMatrix,
    SceneGraph,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    DepthRenderCamera,
    RenderCameraCore,
    CameraInfo,
    ClippingRange,
    DepthRange,
    BsplineTrajectory,
    PositionConstraint,
    KinematicTrajectoryOptimization,
    OrientationConstraint,
    JointStiffnessController,
    PortSwitch,
    AbstractValue,
    InputPortIndex,
)
from pydrake.multibody import inverse_kinematics
from manipulation.scenarios import AddIiwa, AddWsg, AddRgbdSensors, AddPlanarIiwa


def add_iiwa_system(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    iiwa_instance_idx: ModelInstanceIndex,
    iiwa_time_step: float,
) -> tuple:
    """
    Adds an iiwa controller for `iiwa_instance_idx`.

    :returns: A tuple of:
        - iiwa_controller: The iiwa controller.
        - position_input: The position input port.
        - position_commanded_output: The position commanded output port.
        - state_estimated_output: The estimated state output port.
        - position_measured_output: The measured position output port.
        - velocity_estimated_output: The estimated velocity output port.
        - feedforward_torque_input: The feedforward torque input port.
        - torque_commanded_output: The commanded torque output port.
        - torque_external_output: The external torque output port.
    """
    iiwa_instance_name = plant.GetModelInstanceName(iiwa_instance_idx)
    num_iiwa_positions = plant.num_positions(iiwa_instance_idx)

    # Add a PassThrough system for exporting the input port
    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
    position_input = iiwa_position.get_input_port()
    position_commanded_output = iiwa_position.get_output_port()

    # Export the iiwa "state" outputs
    demux = builder.AddSystem(Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions))
    builder.Connect(
        plant.get_state_output_port(iiwa_instance_idx), demux.get_input_port()
    )
    state_estimated_output = plant.get_state_output_port(iiwa_instance_idx)
    position_measured_output = demux.get_output_port(0)
    velocity_estimated_output = demux.get_output_port(1)

    # Make the plant for the iiwa controller to use
    iiwa_controller_plant = MultibodyPlant(time_step=iiwa_time_step)
    iiwa_controller_plant.set_name(iiwa_instance_name + "_controller_plant")
    if plant.num_positions(iiwa_instance_idx) == 3:
        controller_iiwa = AddPlanarIiwa(iiwa_controller_plant)
    else:
        controller_iiwa = AddIiwa(iiwa_controller_plant)
    AddWsg(iiwa_controller_plant, controller_iiwa, welded=True)
    iiwa_controller_plant.Finalize()

    # Add the iiwa controllers
    iiwa_inverse_dynamics_controller = builder.AddSystem(
        InverseDynamicsController(
            iiwa_controller_plant,
            kp=[100] * num_iiwa_positions,
            ki=[1] * num_iiwa_positions,
            kd=[20] * num_iiwa_positions,
            has_reference_acceleration=False,
        )
    )
    iiwa_inverse_dynamics_controller.set_name(
        iiwa_instance_name + "_inverse_dynamics_controller"
    )
    iiwa_stiffness_controller = builder.AddSystem(
        JointStiffnessController(
            iiwa_controller_plant,
            kp=[500] * num_iiwa_positions,
            kd=[30] * num_iiwa_positions,
        )
    )
    iiwa_stiffness_controller.set_name(iiwa_instance_name + "_stiffness_controller")
    builder.Connect(
        plant.get_state_output_port(iiwa_instance_idx),
        iiwa_inverse_dynamics_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        plant.get_state_output_port(iiwa_instance_idx),
        iiwa_stiffness_controller.get_input_port_estimated_state(),
    )

    # Use a ports switch to switch between inverse dynamics and stiffness controllers
    switch = builder.AddSystem(PortSwitch(num_iiwa_positions))
    builder.Connect(
        iiwa_inverse_dynamics_controller.get_output_port_control(),
        switch.DeclareInputPort("inverse_dynamics"),
    )
    builder.Connect(
        iiwa_stiffness_controller.get_output_port_generalized_force(),
        switch.DeclareInputPort("stiffness"),
    )
    # Use a PassThrough system to export the control mode port
    iiwa_control_mode = builder.AddSystem(
        PassThrough(AbstractValue.Make(InputPortIndex(0)))
    )
    iiwa_control_mode_input = iiwa_control_mode.get_input_port()
    builder.Connect(
        iiwa_control_mode.get_output_port(), switch.get_port_selector_input_port()
    )

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(switch.get_output_port(), adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values if not connected)
    torque_passthrough = builder.AddSystem(PassThrough([0] * num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
    builder.Connect(
        adder.get_output_port(), plant.get_actuation_input_port(iiwa_instance_idx)
    )
    feedforward_torque_input = torque_passthrough.get_input_port()

    # Add discrete derivative to command velocities
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_iiwa_positions, iiwa_time_step, suppress_initial_transient=True
        )
    )
    desired_state_from_position.set_name(
        iiwa_instance_name + "_desired_state_from_position"
    )
    builder.Connect(
        desired_state_from_position.get_output_port(),
        iiwa_inverse_dynamics_controller.get_input_port_desired_state(),
    )
    builder.Connect(
        desired_state_from_position.get_output_port(),
        iiwa_stiffness_controller.get_input_port_desired_state(),
    )
    builder.Connect(
        iiwa_position.get_output_port(), desired_state_from_position.get_input_port()
    )

    # Export commanded torques
    torque_commanded_output = adder.get_output_port()
    torque_external_output = plant.get_generalized_contact_forces_output_port(
        iiwa_instance_idx
    )

    return (
        iiwa_controller_plant,
        iiwa_control_mode_input,
        position_input,
        position_commanded_output,
        state_estimated_output,
        position_measured_output,
        velocity_estimated_output,
        feedforward_torque_input,
        torque_commanded_output,
        torque_external_output,
    )


def add_wsg_system(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    wsg_instance_idx: ModelInstanceIndex,
    wsg_time_step: float,
) -> tuple:
    """
    Adds a wsg controller for `wsg_instance_idx`.

    :return: A tuple of:
        - position_input: The commanded position input port.
        - force_limit_input: The force limit input port.
        - state_measured_output: The measured state output port.
        - force_measured_output: The measured force output port.
    """
    wsg_instance_name = plant.GetModelInstanceName(wsg_instance_idx)

    # WSG controller
    wsg_controller = builder.AddSystem(
        SchunkWsgPositionController(time_step=wsg_time_step)
    )
    wsg_controller.set_name(wsg_instance_name + "_controller")

    builder.Connect(
        wsg_controller.get_generalized_force_output_port(),
        plant.get_actuation_input_port(wsg_instance_idx),
    )
    builder.Connect(
        plant.get_state_output_port(wsg_instance_idx),
        wsg_controller.get_state_input_port(),
    )
    position_input = wsg_controller.get_desired_position_input_port()
    force_limit_input = wsg_controller.get_force_limit_input_port()

    # Export state
    wsg_mbp_state_to_wsg_state = builder.AddSystem(MakeMultibodyStateToWsgStateSystem())
    builder.Connect(
        plant.get_state_output_port(wsg_instance_idx),
        wsg_mbp_state_to_wsg_state.get_input_port(),
    )
    state_measured_output = wsg_mbp_state_to_wsg_state.get_output_port()
    force_measured_output = wsg_controller.get_grip_force_output_port()

    return (
        position_input,
        force_limit_input,
        state_measured_output,
        force_measured_output,
    )


def add_cameras(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    width: float,
    height: float,
    fov_y: float,
    camera_prefix: str = "camera",
):
    """
    Adds RGBD cameras for all model intstances in `plant` that start with `camera_prefix`.
    The intrinsics are specified by `width`, `height`, and `fov_y`.
    """
    renderer = "renderer"
    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer, MakeRenderEngineVtk(RenderEngineVtkParams()))
    depth_camera = DepthRenderCamera(
        RenderCameraCore(
            renderer,
            CameraInfo(width=width, height=height, fov_y=fov_y),
            ClippingRange(near=0.1, far=10.0),
            RigidTransform(),
        ),
        DepthRange(0.1, 10.0),
    )
    AddRgbdSensors(
        builder,
        plant,
        scene_graph,
        model_instance_prefix=camera_prefix,
        also_add_point_clouds=False,
        depth_camera=depth_camera,
        renderer=renderer,
    )


def convert_camera_poses_to_iiwa_eef_poses(X_CWs: np.ndarray) -> np.ndarray:
    """
    :param X_CWs: Homogenous world2cam transforms of shape (n,4,4) where n is the number of camera poses. OpenCV
        convention.
    :return: Homogenous X_WG (gripper2world) transforms of shape (n,4,4) where n is the number of poses.
    """
    X_WGs = []
    for X_CW in X_CWs:
        X_WGs.append(np.linalg.inv(X_CW))
    return np.stack(X_WGs)


def calc_inverse_kinematics(
    plant: MultibodyPlant,
    X_G: RigidTransform,
    initial_guess: np.ndarray,
    position_tolerance: float,
    orientation_tolerance: float,
) -> Optional[np.ndarray]:
    """
    :param plant: The iiwa control plant.
    :param X_G: Gripper pose to compute the joint angles for.
    :param initial_guess: The initial guess to use of shape (n,) where n are the number of positions.
    :param position_tolerance: The position tolerance to use for the global IK optimization problem.
    :param orientation_tolerance: The orientation tolerance to use for the global IK optimization problem.
    :return: Joint configuration for the iiwa. Returns None if no IK solution could be found.
    """
    ik = inverse_kinematics.InverseKinematics(plant)
    q_variables = ik.q()

    gripper_frame = plant.GetFrameByName("body")

    # Position constraint
    p_G_ref = X_G.translation()
    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=np.zeros(3),
        frameA=plant.world_frame(),
        p_AQ_lower=p_G_ref - position_tolerance,
        p_AQ_upper=p_G_ref + position_tolerance,
    )

    # Orientation constraint
    R_G_ref = X_G.rotation()
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=R_G_ref,
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix(),
        theta_bound=orientation_tolerance,
    )

    prog = ik.prog()
    prog.SetInitialGuess(q_variables, initial_guess)

    result = Solve(prog)
    if not result.is_success():
        return None
    q_sol = result.GetSolution(q_variables)
    return q_sol


class IIWAJointTrajectorySource(LeafSystem):
    """Computes a trajectory between two poses in joint space."""

    def __init__(
        self,
        plant: MultibodyPlant,
        q_nominal: np.ndarray,
        t_start: float = 0.0,
        iiwa_num_positions: int = 7,
        meshcat: Optional[Meshcat] = None,
    ):
        """
        :param plant: The iiwa control plant.
        :param q_nominal: The iiwa initial joint positions.
        :param t_start: The trajectory start time.
        :param iiwa_num_positions: The number of joints of the iiwa contained in `plant`.
        :param meshcat: A meshcat visualizer.
        """
        super().__init__()

        self._plant = plant
        self._q_nominal = q_nominal
        self._t_start = t_start
        self._iiwa_num_position = iiwa_num_positions
        self._meshcat = meshcat

        self._q_knots = []  # Joint angles at knots
        self._breaks = []  # Times between knots
        self._q_traj = None

        # iiwa desired state [q, q_dot]
        self.x_output_port = self.DeclareVectorOutputPort(
            "traj_x", BasicVector(self._iiwa_num_position * 2), self._calc_x
        )

    def set_meshcat(self, meshcat: Meshcat) -> None:
        """Replaces the meshcat visualizer."""
        self._meshcat = meshcat

    def set_trajectory(self, q_traj: PiecewisePolynomial, t_start: float = 0.0) -> None:
        """
        :param q_traj: The trajectory to set.
        :param t_start: The trajectory start time.
        """
        self._q_traj = q_traj
        self._t_start = t_start

    def compute_and_set_trajectory(
        self,
        X_WGs: List[RigidTransform],
        time_between_breakpoints: Union[float, List[float]],
        ik_position_tolerance: float,
        ik_orientation_tolerance: float,
        allow_no_ik_sols: bool,
        debug: bool = False,
    ) -> PiecewisePolynomial:
        """
        Computes and sets a new trajectory.
        NOTE: We need to reset the state using `set_t_start` before we call this method a second time.

        :param X_WG: A path of eef poses.
        :param time_between_breakpoints: The time between the poses in the path. If a float, the same duration is used
            for all pose pairs. If a list, then the list must contain one time value for each pose. The first value is
            the time taken to reach the first pose.
        :param ik_position_tolerance: The position tolerance to use for the global IK optimization problem.
        :param ik_orientation_tolerance: The orientation tolerance to use for the global IK optimization problem.
        :param allow_no_ik_sols: If true, don't raise an exception on no IK solution found but skip to the next one.
        :param debug: If true, visualize the path in meshcat.
        :return: The joint trajectory.
        """
        assert isinstance(time_between_breakpoints, float) or len(
            time_between_breakpoints
        ) == len(X_WGs)

        t = 0.0
        for i, X_WG in enumerate(X_WGs):
            # Use the eef pose at the previous knot point as an initial guess
            initial_guess = (
                self._q_nominal if len(self._q_knots) == 0 else self._q_knots[-1]
            )
            q = calc_inverse_kinematics(
                self._plant,
                X_WG,
                initial_guess,
                ik_position_tolerance,
                ik_orientation_tolerance,
            )
            if q is None:
                if allow_no_ik_sols:
                    print(
                        f"set_trajectory: No IK solution could be found for {i}th pose in path. Skipping to next pose."
                    )
                    continue
                else:
                    raise RuntimeError(
                        f"No IK solution found for {i}th pose. Translation: {X_WG.translation()}, Rotation:\n{X_WG.rotation()}"
                    )

            self._q_knots.append(q)
            self._breaks.append(t)

            t += (
                time_between_breakpoints
                if isinstance(time_between_breakpoints, float)
                else time_between_breakpoints[i]
            )

            if debug and self._meshcat is not None:
                AddMeshcatTriad(
                    self._meshcat, f"X_WG{i}", length=0.15, radius=0.006, X_PT=X_WG
                )
        self._q_traj = self._calc_q_traj()

        return self._q_traj

    def _calc_x(self, context, output) -> None:
        """
        :return: The robot state [q, q_dot] where q are the joint positions.
        """
        if self._q_traj:
            t = context.get_time() - self._t_start
            q = self._q_traj.value(t).ravel()
            q_dot = self._q_traj.derivative(1).value(t).ravel()
        else:
            q = self._q_nominal
            q_dot = np.array([0.0] * self._iiwa_num_position)
        output.SetFromVector(np.hstack([q, q_dot]))

    def set_t_start(self, t_start_new: float) -> None:
        self._t_start = t_start_new
        self._q_knots = []
        self._breaks = []

    def get_q_traj(self) -> PiecewisePolynomial:
        return self._q_traj

    def _calc_q_traj(self) -> PiecewisePolynomial:
        """
        Generate a joint configuration trajectory from a beginning and end configuration.

        :return: PiecewisePolynomial
        """
        assert len(self._q_knots) == len(self._breaks)
        return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            self._breaks,
            np.vstack(self._q_knots).T,
            np.zeros(self._iiwa_num_position),
            np.zeros(self._iiwa_num_position),
        )


class IIWAOptimizedJointTrajectorySource(LeafSystem):
    """Computes a trajectory for a path using kinematic trajectory optimization."""

    def __init__(
        self,
        iiwa_control_plant: MultibodyPlant,
        q_nominal: np.ndarray,
        t_start: float = 0.0,
        iiwa_num_positions: int = 7,
    ):
        """
        :param iiwa_control_plant: The iiwa control plant.
        :param q_nominal: The iiwa initial joint positions.
        :param t_start: The trajectory start time.
        :param iiwa_num_positions: The number of joints of the iiwa contained in `plant`.
        """
        super().__init__()

        self._iiwa_control_plant = iiwa_control_plant
        self._q_nominal = q_nominal
        self._t_start = t_start
        self._iiwa_num_position = iiwa_num_positions

        self._q_traj = None

        # iiwa desired state [q, q_dot]
        self.x_output_port = self.DeclareVectorOutputPort(
            "traj_x", BasicVector(self._iiwa_num_position * 2), self._calc_x
        )

    def set_trajectory(
        self,
        X_WGs: List[RigidTransform],
        initial_guess: np.ndarray,
        ik_position_tolerance: float,
        ik_orientation_tolerance: float,
    ) -> float:
        """
        Used to set a new trajectory using kinematic trajectory optimization.

        :param X_WG: The waypoints that the eef must go through (added as position constraints).
        :param initial_guess: The initial joint position guess of shape (7, n) where n is the number of control points.
        :param ik_position_tolerance: The position tolerance to use for the global IK optimization problem.
        :param ik_orientation_tolerance: The orientation tolerance to use for the global IK optimization problem.
        :param allow_no_ik_sols: If true, don't raise an exception on no IK solution found but skip to the next one.
        :return: The trajectory end time.
        """
        num_control_points = initial_guess.shape[1]
        plant_context = self._iiwa_control_plant.CreateDefaultContext()
        num_q = self._iiwa_control_plant.num_positions()
        gripper_frame = self._iiwa_control_plant.GetFrameByName("body")

        trajopt = KinematicTrajectoryOptimization(
            num_positions=self._iiwa_control_plant.num_positions(),
            num_control_points=num_control_points,
        )
        prog = trajopt.get_mutable_prog()

        # Set initial guess
        path_guess = BsplineTrajectory(trajopt.basis(), initial_guess)
        trajopt.SetInitialGuess(path_guess)

        # Add iiwa limits
        trajopt.AddPositionBounds(
            self._iiwa_control_plant.GetPositionLowerLimits(),
            self._iiwa_control_plant.GetPositionUpperLimits(),
        )
        # Enforce slow velocities
        trajopt.AddVelocityBounds(
            self._iiwa_control_plant.GetVelocityLowerLimits(),
            self._iiwa_control_plant.GetVelocityUpperLimits() / 5.0,
        )

        # TODO: This should be an argument
        trajopt.AddDurationConstraint(lb=2.0, ub=30.0)

        # Add pose constraints
        for i, s in enumerate(np.linspace(0, 1, len(X_WGs))):
            translation_constraint = PositionConstraint(
                plant=self._iiwa_control_plant,
                frameA=self._iiwa_control_plant.world_frame(),
                p_AQ_lower=X_WGs[i].translation() - ik_position_tolerance,
                p_AQ_upper=X_WGs[i].translation() + ik_position_tolerance,
                frameB=gripper_frame,
                p_BQ=[0, 0, 0],
                plant_context=plant_context,
            )
            trajopt.AddPathPositionConstraint(constraint=translation_constraint, s=s)

            orientation_constraint = OrientationConstraint(
                plant=self._iiwa_control_plant,
                frameAbar=self._iiwa_control_plant.world_frame(),
                R_AbarA=X_WGs[i].rotation(),
                frameBbar=gripper_frame,
                R_BbarB=RotationMatrix(),
                theta_bound=ik_orientation_tolerance,
                plant_context=plant_context,
            )
            trajopt.AddPathPositionConstraint(constraint=orientation_constraint, s=s)

        # start and end with zero velocity
        trajopt.AddPathVelocityConstraint(
            lb=np.zeros((num_q, 1)), ub=np.zeros((num_q, 1)), s=0
        )
        trajopt.AddPathVelocityConstraint(
            lb=np.zeros((num_q, 1)), ub=np.zeros((num_q, 1)), s=1
        )

        trajopt.AddDurationCost(1.0)

        result = Solve(prog)
        if not result.is_success():
            print("Trajectory optimization failed. Executing trajectory anyways.")

        self._q_traj = trajopt.ReconstructTrajectory(result)

        return self._q_traj.end_time()

    def _calc_x(self, context, output) -> None:
        """
        :return: The robot state [q, q_dot] where q are the joint positions.
        """
        if self._q_traj:
            t = context.get_time() - self._t_start
            q = self._q_traj.value(t).ravel()
            q_dot = self._q_traj.MakeDerivative(1).value(t).ravel()
        else:
            q = self._q_nominal
            q_dot = np.array([0.0] * self._iiwa_num_position)
        output.SetFromVector(np.hstack([q, q_dot]))

    def set_t_start(self, t_start_new: float) -> None:
        self._t_start = t_start_new


class WSGCommandSource(LeafSystem):
    """Commands the WSG to the last specified position."""

    def __init__(self, initial_pos: float):
        """
        :param initial_pos: The angle to command the table to.
        """
        super().__init__()

        self._command_pos = initial_pos

        self._control_output_port = self.DeclareVectorOutputPort(
            "wsg_position", 1, self.CalcOutput
        )

    def set_new_pos_command(self, pos: float) -> None:
        """
        :param pos: The new WSG position to command.
        """
        self._command_pos = pos

    def CalcOutput(self, context, output):
        output.SetAtIndex(0, self._command_pos)


def prune_infeasible_eef_poses(
    X_WGs: np.ndarray,
    plant: MultibodyPlant,
    initial_guess: np.ndarray,
    ik_position_tolerance: float,
    ik_orientation_tolerance: float,
) -> np.ndarray:
    """
    Removes all poses for which no iiwa IK can be found.

    :param X_WG: The eef waypoints to prune.
    :param initial_guess: The initial joint position guess of shape (7,).
    :param ik_position_tolerance: The position tolerance to use for the global IK optimization problem.
    :param ik_orientation_tolerance: The orientation tolerance to use for the global IK optimization problem.
    :return: The pruned waypoints.
    """
    X_WG_feasible = []
    for X_WG in X_WGs:
        sol = calc_inverse_kinematics(
            plant,
            RigidTransform(X_WG),
            initial_guess,
            ik_position_tolerance,
            ik_orientation_tolerance,
        )
        if sol is not None:
            X_WG_feasible.append(X_WG)
            initial_guess = sol
    return np.stack(X_WG_feasible)


class IIWAControlModeSource(LeafSystem):
    class ControllerMode(Enum):
        INVERSE_DYNAMICS = 0
        STIFFNESS = 1

    def __init__(self):
        super().__init__()

        self._mode = self.ControllerMode.INVERSE_DYNAMICS

        self.DeclareAbstractOutputPort(
            "iiwa_control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode,
        )

    def set_control_mode(
        self, control_mode: "IIWAControlModeSource.ControllerMode"
    ) -> None:
        self._mode = control_mode

    def CalcControlMode(self, context, output):
        # mode = context.get_abstract_state(int(self._mode_index)).get_value()

        output.set_value(InputPortIndex(2))

        if self._mode == self.ControllerMode.INVERSE_DYNAMICS:
            # inverse dynamics controller
            output.set_value(InputPortIndex(1))
        else:
            # stiffness controller
            output.set_value(InputPortIndex(2))
