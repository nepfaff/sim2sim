from typing import List

import numpy as np

from pydrake.all import ExternallyAppliedSpatialForce, LeafSystem, SpatialForce, Value


class ExternalForceSystem(LeafSystem):
    """A system for applying a force to a body."""

    def __init__(self, target_body_idx):
        LeafSystem.__init__(self)

        self._target_body_index = target_body_idx

        self.DeclareAbstractOutputPort(
            "spatial_forces_vector",
            lambda: Value[List[ExternallyAppliedSpatialForce]](),
            self.DoCalcAbstractOutput,
        )

        self._wrench = np.zeros(6)
        self._wrench_application_point = np.zeros(3)

    def DoCalcAbstractOutput(self, context, output):
        force = ExternallyAppliedSpatialForce()
        force.body_index = self._target_body_index
        force.p_BoBq_B = self._wrench_application_point
        force.F_Bq_W = SpatialForce(tau=self._wrench[3:], f=self._wrench[:3])
        output.set_value([force])

    def set_wrench(self, wrench: np.ndarray) -> None:
        self._wrench = wrench

    def set_wrench_application_point(self, point: np.ndarray) -> None:
        self._wrench_application_point = point


class StateSource(LeafSystem):
    """
    A system for commanding desired states (q, q_dot).
    """

    def __init__(self, starting_position: List[float]):
        """
        :param starting_position: The sphere starting position [x, y, z].
        """
        LeafSystem.__init__(self)

        self.DeclareVectorOutputPort("desired_state", 6, self.CalcOutput)  # q, q_dot

        self._desired_state = [*starting_position, 0.0, 0.0, 0.0]

    def CalcOutput(self, context, output):
        output.SetFromVector(self._desired_state)

    def set_desired_state(self, desired_state: List[float]) -> None:
        """
        :param desired_state: The desired state [x, y, z, x_dot, y_dot, z_dot].
        """
        self._desired_state = desired_state

    def set_desired_position(self, desired_position: List[float]) -> None:
        """
        NOTE: Desired velocities will be set to zero.

        :param desired position: The desired position [x, y, z].
        """
        self._desired_state = [*desired_position, 0.0, 0.0, 0.0]
