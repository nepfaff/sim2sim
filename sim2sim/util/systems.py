from typing import List

import numpy as np
from pydrake.all import LeafSystem, Value, ExternallyAppliedSpatialForce, SpatialForce


class ExternalForceSystem(LeafSystem):
    """A system for applying a force to a body."""

    def __init__(self, target_body_idx):
        LeafSystem.__init__(self)

        self._target_body_index = target_body_idx

        self.DeclareAbstractOutputPort(
            "spatial_forces_vector", lambda: Value[List[ExternallyAppliedSpatialForce]](), self.DoCalcAbstractOutput
        )

        self._wrench = np.zeros(6)
        self._wrench_application_point = np.zeros(3)

    def DoCalcAbstractOutput(self, context, output):
        test_force = ExternallyAppliedSpatialForce()
        test_force.body_index = self._target_body_index
        test_force.p_BoBq_B = self._wrench_application_point
        test_force.F_Bq_W = SpatialForce(tau=self._wrench[3:], f=self._wrench[:3])
        output.set_value([test_force])

    def set_wrench(self, wrench: np.ndarray) -> None:
        self._wrench = wrench

    def set_wrench_application_point(self, point: np.ndarray) -> None:
        self._wrench_application_point = point
