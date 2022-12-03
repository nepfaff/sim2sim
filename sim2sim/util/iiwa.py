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
    Diagram,
)
from pydrake.multibody import inverse_kinematics
from manipulation.scenarios import AddIiwa, AddWsg, AddRgbdSensors, AddPlanarIiwa

# Transform from iiwa link 7 to eef (gripper)
X_L7E = RigidTransform(RollPitchYaw(np.pi / 2, 0, np.pi / 2), np.array([0, 0, 0.09]))


def make_iiwa_wsg_system(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    iiwa_time_step: float,
    iiwa_prefix: str = "iiwa",
    wsg_prefix: str = "wsg",
    camera_prefix: str = "camera",
) -> Diagram:
    """
    Iterates through all model instances in `plant`:
    - For each model instance starting with `iiwa_prefix`, we add an additional iiwa controller system
    - For each model instance starting with `wsg_prefix`, we add an additional schunk controller system
    - For each body starting with `camera_prefix`, we add a RgbdSensor

    :param builder: The diagram builder.
    :param plant: The finalized plant.
    :param scene_graph: The scene graph.
    :param iiwa_time_step: The iiwa control plant timestep.
    :param iiwa_prefix: Any model instances starting with `iiwa_prefix` will get an inverse dynamics controller, etc.
        attached
    :param wsg_prefix: Any model instance starting with `wsg_prefix` will get a schunk controller
    :param camera_prefix: Any bodies in the plant (created during the plant_setup_callback) starting with this prefix
        will get a camera attached.
    """
    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)

        if model_instance_name.startswith(iiwa_prefix):
            num_iiwa_positions = plant.num_positions(model_instance)

            # Add a PassThrough system for exporting the input port
            iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
            builder.ExportInput(iiwa_position.get_input_port(), model_instance_name + "_position")
            builder.ExportOutput(iiwa_position.get_output_port(), model_instance_name + "_position_commanded")

            # Export the iiwa "state" outputs
            demux = builder.AddSystem(Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions))
            builder.Connect(plant.get_state_output_port(model_instance), demux.get_input_port())
            builder.ExportOutput(demux.get_output_port(0), model_instance_name + "_position_measured")
            builder.ExportOutput(demux.get_output_port(1), model_instance_name + "_velocity_estimated")
            builder.ExportOutput(plant.get_state_output_port(model_instance), model_instance_name + "_state_estimated")

            # Make the plant for the iiwa controller to use
            iiwa_controller_plant = MultibodyPlant(time_step=iiwa_time_step)
            if plant.num_positions(model_instance) == 3:
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
            iiwa_controller.set_name(model_instance_name + "_controller")
            builder.Connect(
                plant.get_state_output_port(model_instance), iiwa_controller.get_input_port_estimated_state()
            )

            # Add in the feed-forward torque
            adder = builder.AddSystem(Adder(2, num_iiwa_positions))
            builder.Connect(iiwa_controller.get_output_port_control(), adder.get_input_port(0))
            # Use a PassThrough to make the port optional (it will provide zero values if not connected)
            torque_passthrough = builder.AddSystem(PassThrough([0] * num_iiwa_positions))
            builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
            builder.ExportInput(torque_passthrough.get_input_port(), model_instance_name + "_feedforward_torque")
            builder.Connect(adder.get_output_port(), plant.get_actuation_input_port(model_instance))

            # Add discrete derivative to command velocities
            desired_state_from_position = builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(
                    num_iiwa_positions, iiwa_time_step, suppress_initial_transient=True
                )
            )
            desired_state_from_position.set_name(model_instance_name + "_desired_state_from_position")
            builder.Connect(
                desired_state_from_position.get_output_port(), iiwa_controller.get_input_port_desired_state()
            )
            builder.Connect(iiwa_position.get_output_port(), desired_state_from_position.get_input_port())

            # Export commanded torques
            builder.ExportOutput(adder.get_output_port(), model_instance_name + "_torque_commanded")
            builder.ExportOutput(adder.get_output_port(), model_instance_name + "_torque_measured")
            builder.ExportOutput(
                plant.get_generalized_contact_forces_output_port(model_instance),
                model_instance_name + "_torque_external",
            )

        elif model_instance_name.startswith(wsg_prefix):
            # Wsg controller.
            wsg_controller = builder.AddSystem(SchunkWsgPositionController())
            wsg_controller.set_name(model_instance_name + "_controller")
            builder.Connect(
                wsg_controller.get_generalized_force_output_port(), plant.get_actuation_input_port(model_instance)
            )
            builder.Connect(plant.get_state_output_port(model_instance), wsg_controller.get_state_input_port())
            builder.ExportInput(wsg_controller.get_desired_position_input_port(), model_instance_name + "_position")
            builder.ExportInput(wsg_controller.get_force_limit_input_port(), model_instance_name + "_force_limit")
            wsg_mbp_state_to_wsg_state = builder.AddSystem(MakeMultibodyStateToWsgStateSystem())
            builder.Connect(plant.get_state_output_port(model_instance), wsg_mbp_state_to_wsg_state.get_input_port())
            builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(), model_instance_name + "_state_measured")
            builder.ExportOutput(wsg_controller.get_grip_force_output_port(), model_instance_name + "_force_measured")

    # Add cameras
    AddRgbdSensors(builder, plant, scene_graph, model_instance_prefix=camera_prefix)

    # Export "cheat" ports
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
    builder.ExportOutput(plant.get_state_output_port(), "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("iiwa_wsg_system")
    return diagram


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

    def _calc_x(self, context, output):
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

    def set_t_start(self, t_start_new: float):
        self._t_start = t_start_new

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
