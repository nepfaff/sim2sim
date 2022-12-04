from typing import List, Union, Optional

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
    RollPitchYaw,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    DepthRenderCamera,
    RenderCameraCore,
    CameraInfo,
    ClippingRange,
    DepthRange,
)
from pydrake.multibody import inverse_kinematics
from manipulation.scenarios import AddIiwa, AddWsg, AddRgbdSensors, AddPlanarIiwa

# Transform from iiwa link 7 to eef (gripper)
X_L7E = RigidTransform(RollPitchYaw(np.pi / 2, 0, np.pi / 2), np.array([0, 0, 0.09]))


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
    builder.Connect(plant.get_state_output_port(iiwa_instance_idx), demux.get_input_port())
    state_estimated_output = plant.get_state_output_port(iiwa_instance_idx)
    position_measured_output = demux.get_output_port(0)
    velocity_estimated_output = demux.get_output_port(1)

    # Make the plant for the iiwa controller to use
    iiwa_controller_plant = MultibodyPlant(time_step=iiwa_time_step)
    if plant.num_positions(iiwa_instance_idx) == 3:
        controller_iiwa = AddPlanarIiwa(iiwa_controller_plant)
    else:
        controller_iiwa = AddIiwa(iiwa_controller_plant)
    AddWsg(iiwa_controller_plant, controller_iiwa, welded=True)
    iiwa_controller_plant.Finalize()

    # Add the iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            iiwa_controller_plant,
            kp=[100] * num_iiwa_positions,
            ki=[1] * num_iiwa_positions,
            kd=[20] * num_iiwa_positions,
            has_reference_acceleration=False,
        )
    )
    iiwa_controller.set_name(iiwa_instance_name + "_controller")
    builder.Connect(plant.get_state_output_port(iiwa_instance_idx), iiwa_controller.get_input_port_estimated_state())

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(), adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values if not connected)
    torque_passthrough = builder.AddSystem(PassThrough([0] * num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
    builder.Connect(adder.get_output_port(), plant.get_actuation_input_port(iiwa_instance_idx))
    feedforward_torque_input = torque_passthrough.get_input_port()

    # Add discrete derivative to command velocities
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(num_iiwa_positions, iiwa_time_step, suppress_initial_transient=True)
    )
    desired_state_from_position.set_name(iiwa_instance_name + "_desired_state_from_position")
    builder.Connect(desired_state_from_position.get_output_port(), iiwa_controller.get_input_port_desired_state())
    builder.Connect(iiwa_position.get_output_port(), desired_state_from_position.get_input_port())

    # Export commanded torques
    torque_commanded_output = adder.get_output_port()
    torque_external_output = plant.get_generalized_contact_forces_output_port(iiwa_instance_idx)

    return (
        iiwa_controller,
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
    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    wsg_controller.set_name(wsg_instance_name + "_controller")

    builder.Connect(
        wsg_controller.get_generalized_force_output_port(), plant.get_actuation_input_port(wsg_instance_idx)
    )
    builder.Connect(plant.get_state_output_port(wsg_instance_idx), wsg_controller.get_state_input_port())
    position_input = wsg_controller.get_desired_position_input_port()
    force_limit_input = wsg_controller.get_force_limit_input_port()

    # Export state
    wsg_mbp_state_to_wsg_state = builder.AddSystem(MakeMultibodyStateToWsgStateSystem())
    builder.Connect(plant.get_state_output_port(wsg_instance_idx), wsg_mbp_state_to_wsg_state.get_input_port())
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

    def set_trajectory(
        self,
        X_WGs: List[RigidTransform],
        time_between_breakpoints: Union[float, List[float]],
        ik_position_tolerance: float,
        ik_orientation_tolerance: float,
        allow_no_ik_sols: bool,
        debug: bool,
    ):
        """
        Used to set a new trajectory.

        :param X_WG: A path of eef poses.
        :param time_between_breakpoints: The time between the poses in the path. If a float, the same duration is used
            for all pose pairs. If a list, then the list must contain one time value for each pose. The first value is
            the time taken to reach the first pose.
        :param ik_position_tolerance: The position tolerance to use for the global IK optimization problem.
        :param ik_orientation_tolerance: The orientation tolerance to use for the global IK optimization problem.
        :param allow_no_ik_sols: If true, don't raise an exception on no IK solution found but skip to the next one.
        :param debug: If true, visualize the path in meshcat.
        """
        assert isinstance(time_between_breakpoints, float) or len(time_between_breakpoints) == len(X_WGs)

        t = 0.0
        for i, X_WG in enumerate(X_WGs):
            q = self._inverse_kinematics(X_WG, ik_position_tolerance, ik_orientation_tolerance)
            if q is None:
                if allow_no_ik_sols:
                    print(
                        f"set_trajectory: No IK solution could be found for {i}th pose in path. Skipping to next pose."
                    )
                    continue
                else:
                    raise RuntimeError(f"No IK solution found for {i}th pose.")

            self._q_knots.append(q)
            self._breaks.append(t)

            t += (
                time_between_breakpoints if isinstance(time_between_breakpoints, float) else time_between_breakpoints[i]
            )

            if debug and self._meshcat is not None:
                AddMeshcatTriad(self._meshcat, f"X_WG{i}", length=0.15, radius=0.006, X_PT=X_WG)
        self._q_traj = self._calc_q_traj()

    def _inverse_kinematics(
        self,
        X_G: RigidTransform,
        position_tolerance: float,
        orientation_tolerance: float,
    ) -> Optional[np.ndarray]:
        """
        :param X_G: Gripper pose to compute the joint angles for.
        :param position_tolerance: The position tolerance to use for the global IK optimization problem.
        :param orientation_tolerance: The orientation tolerance to use for the global IK optimization problem.
        :return: Joint configuration for the iiwa. Returns None if no IK solution could be found.
        """
        ik = inverse_kinematics.InverseKinematics(self._plant)
        q_variables = ik.q()

        gripper_frame = self._plant.GetFrameByName("body")

        # Position constraint
        p_G_ref = X_G.translation()
        ik.AddPositionConstraint(
            frameB=gripper_frame,
            p_BQ=np.zeros(3),
            frameA=self._plant.world_frame(),
            p_AQ_lower=p_G_ref - position_tolerance,
            p_AQ_upper=p_G_ref + position_tolerance,
        )

        # Orientation constraint
        R_G_ref = X_G.rotation()
        ik.AddOrientationConstraint(
            frameAbar=self._plant.world_frame(),
            R_AbarA=R_G_ref,
            frameBbar=gripper_frame,
            R_BbarB=RotationMatrix(),
            theta_bound=orientation_tolerance,
        )

        prog = ik.prog()

        # Use the eef pose at the previous knot point as an initial guess
        if len(self._q_knots) == 0:
            init_guess = self._q_nominal
        else:
            init_guess = self._q_knots[-1]
        prog.SetInitialGuess(q_variables, init_guess)

        result = Solve(prog)
        if not result.is_success():
            return None
        q_sol = result.GetSolution(q_variables)
        return q_sol

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
