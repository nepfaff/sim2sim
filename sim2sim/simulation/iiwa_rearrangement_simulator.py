from typing import List

import numpy as np

from pydrake.all import DiagramBuilder, RigidTransform, RollPitchYaw, SceneGraph

from sim2sim.logging import DynamicLogger
from sim2sim.simulation import IIWAJointTrajectorySimulatorBase
from sim2sim.util import IIWAControlModeSource


class IIWARearrangementSimulator(IIWAJointTrajectorySimulatorBase):
    """The simulator for the iiwa rearrangement task."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLogger,
        is_hydroelastic: bool,
        mesh_pose: List[float],
        skip_outer_visualization: bool = False,
    ):
        """
        :param outer_builder: Diagram builder for the outer simulation environment.
        :param outer_scene_graph: Scene graph for the outer simulation environment.
        :param inner_builder: Diagram builder for the inner simulation environment.
        :param inner_scene_graph: Scene graph for the inner simulation environment.
        :param logger: The logger.
        :param is_hydroelastic: Whether hydroelastic or point contact is used.
        :param mesh_pose: The manipuland mesh pose of form [roll, pitch, yaw, x, y, z] where angles are in radians.
        """
        super().__init__(
            outer_builder,
            outer_scene_graph,
            inner_builder,
            inner_scene_graph,
            logger,
            is_hydroelastic,
            mesh_pose,
            skip_outer_visualization,
        )

    def simulate(self, duration: float) -> None:
        # Use inner diagram for planning
        plant = self._inner_diagram.GetSubsystemByName("plant")
        world_frame = plant.world_frame()
        gripper_frame = plant.GetFrameByName("body")

        X_WG_initial = plant.CalcRelativeTransform(
            plant.CreateDefaultContext(), frame_A=world_frame, frame_B=gripper_frame
        )

        command_sequence = [
            {
                "iiwa_path": [
                    X_WG_initial,
                    RigidTransform(
                        RollPitchYaw(-np.pi / 2, 0.0, -np.pi),
                        self._mesh_pose[3:] + [0.0, 0.0, 0.3],
                    ),
                    RigidTransform(
                        RollPitchYaw(-np.pi / 2, 0.0, -np.pi),
                        self._mesh_pose[3:] + [0.0, 0.0, 0.20],
                    ),
                    RigidTransform(
                        RollPitchYaw(-np.pi / 2, 0.0, -np.pi),
                        self._mesh_pose[3:] + [0.0, 0.0, 0.15],
                    ),
                ],
                "iiwa_control_mode": IIWAControlModeSource.ControllerMode.INVERSE_DYNAMICS,
                "time_between_breakpoints": 0.5,
                "wsg_position": -1.0,
                "time_to_simulate_after": 0.0,
            },
            {
                "iiwa_path": [
                    RigidTransform(
                        RollPitchYaw(-np.pi / 2, 0.0, -np.pi),
                        self._mesh_pose[3:] + [0.0, 0.0, 0.15],
                    ),
                    RigidTransform(
                        RollPitchYaw(-np.pi / 2, 0.0, -np.pi),
                        self._mesh_pose[3:] + [0.0, 0.0, 0.3],
                    ),
                    RigidTransform(
                        RollPitchYaw(-np.pi / 2, 0.0, -np.pi),
                        self._mesh_pose[3:] + [0.0, 0.0, 0.4],
                    ),
                    RigidTransform(
                        RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [-0.25, 0.25, 0.4]
                    ),
                    RigidTransform(
                        RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [-0.5, 0.0, 0.5]
                    ),
                    RigidTransform(
                        RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [-0.5, 0.0, 0.3]
                    ),
                ],
                "iiwa_control_mode": IIWAControlModeSource.ControllerMode.INVERSE_DYNAMICS,
                "time_between_breakpoints": 0.5,
                "wsg_position": 0.1,
                "time_to_simulate_after": 2.0,
            },
        ]

        self._simulate(command_sequence)
