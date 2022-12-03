from typing import List, Union, Optional

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.all import (
    MultibodyPlant,
    PiecewisePolynomial,
    RigidTransform,
    Solve,
    Meshcat,
)
from pydrake.math import RotationMatrix
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.multibody import inverse_kinematics


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
