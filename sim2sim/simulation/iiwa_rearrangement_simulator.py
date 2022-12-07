import os
from typing import List

import numpy as np
from pydrake.all import (
    DiagramBuilder,
    SceneGraph,
    Simulator,
    RigidTransform,
    RollPitchYaw,
)

from sim2sim.logging import DynamicLoggerBase
from sim2sim.simulation import SimulatorBase
from sim2sim.util import IIWAJointTrajectorySource


class IIWARearrangementSimulator(SimulatorBase):
    """The simulator for the iiwa rearrangement scene."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLoggerBase,
        mesh_pose: List[float],
    ):
        """
        :param outer_builder: Diagram builder for the outer simulation environment.
        :param outer_scene_graph: Scene graph for the outer simulation environment.
        :param inner_builder: Diagram builder for the inner simulation environment.
        :param inner_scene_graph: Scene graph for the inner simulation environment.
        :param logger: The logger.
        :param mesh_pose: The manipuland mesh pose of form [roll, pitch, yaw, x, y, z] where angles are in radians.
        """

        super().__init__(outer_builder, outer_scene_graph, inner_builder, inner_scene_graph, logger)
        self._finalize_and_build_diagrams()

        self._mesh_pose = np.array(mesh_pose)

    def _finalize_and_build_diagrams(self) -> None:
        """Adds visualization systems to the outer and inner diagrams and builds them."""

        self._outer_visualizer, self._outer_meshcat = self._logger.add_visualizers(
            self._outer_builder,
            self._outer_scene_graph,
            is_outer=True,
        )
        self._inner_visualizer, self._inner_meshcat = self._logger.add_visualizers(
            self._inner_builder,
            self._inner_scene_graph,
            is_outer=False,
        )

        self._logger.add_pose_logging(self._outer_builder, self._inner_builder)

        self._outer_diagram = self._outer_builder.Build()
        self._inner_diagram = self._inner_builder.Build()

    def _get_command_sequence(self) -> List[dict]:
        # Use inner diagram for planning
        plant = self._inner_diagram.GetSubsystemByName("plant")
        world_frame = plant.world_frame()
        gripper_frame = plant.GetFrameByName("body")
        iiwa_trajectory_source: IIWAJointTrajectorySource = self._inner_diagram.GetSubsystemByName(
            "iiwa_joint_trajectory_source"
        )

        X_WG_initial = plant.CalcRelativeTransform(
            plant.CreateDefaultContext(), frame_A=world_frame, frame_B=gripper_frame
        )

        command_sequence = [
            {
                "iiwa_path": [
                    X_WG_initial,
                    RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), self._mesh_pose[3:] + [0.0, 0.0, 0.3]),
                    RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), self._mesh_pose[3:] + [0.0, 0.0, 0.20]),
                    RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), self._mesh_pose[3:] + [0.0, 0.0, 0.15]),
                ],
                "time_between_breakpoints": 0.5,
                "wsg_position": -1.0,
                "time_to_simulate_after": 0.0,
            },
            {
                "iiwa_path": [
                    RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), self._mesh_pose[3:] + [0.0, 0.0, 0.15]),
                    RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), self._mesh_pose[3:] + [0.0, 0.0, 0.3]),
                    RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), self._mesh_pose[3:] + [0.0, 0.0, 0.4]),
                    RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [0.25, 0.25, 0.4]),
                    RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [0.5, 0.0, 0.5]),
                    RigidTransform(RollPitchYaw(-np.pi / 2, 0.0, -np.pi), [0.5, 0.0, 0.3]),
                ],
                "time_between_breakpoints": 0.5,
                "wsg_position": 0.1,
                "time_to_simulate_after": 2.0,
            },
        ]

        # Add lower level info to command sequence
        low_level_command_sequence = []
        for command in command_sequence:

            # Compute iiwa trajectory
            iiwa_trajectory_source.set_t_start(0.0)
            iiwa_traj = iiwa_trajectory_source.compute_and_set_trajectory(
                command["iiwa_path"],
                time_between_breakpoints=command["time_between_breakpoints"],
                ik_position_tolerance=0.005,
                ik_orientation_tolerance=0.005,
                allow_no_ik_sols=False,
            )

            low_level_command_sequence.append(
                {
                    "iiwa_traj": iiwa_traj,
                    "wsg_position": command["wsg_position"],
                    "time_to_simulate_after": command["time_to_simulate_after"],
                }
            )

        return low_level_command_sequence

    def simulate(self, duration: float) -> None:
        command_sequence = self._get_command_sequence()

        for i, (diagram, visualizer, meshcat) in enumerate(
            zip(
                [self._outer_diagram, self._inner_diagram],
                [self._outer_visualizer, self._inner_visualizer],
                [self._outer_meshcat, self._inner_meshcat],
            )
        ):
            # Get required systems
            plant = diagram.GetSubsystemByName("plant")
            iiwa_trajectory_source: IIWAJointTrajectorySource = diagram.GetSubsystemByName(
                "iiwa_joint_trajectory_source"
            )
            wsg_command_source = diagram.GetSubsystemByName("wsg_command_source")

            # Create simulator
            simulator = Simulator(diagram)
            simulator.Initialize()
            visualizer.StartRecording()
            context = simulator.get_mutable_context()

            for command in command_sequence:
                # Execute iiwa trajectory
                iiwa_trajectory_source.set_trajectory(command["iiwa_traj"], context.get_time())
                simulator.AdvanceTo(context.get_time() + command["iiwa_traj"].end_time())

                # Command wsg
                wsg_command_source.set_new_pos_command(command["wsg_position"])
                simulator.AdvanceTo(context.get_time() + 0.1)

                # Simulate for some more
                simulator.AdvanceTo(context.get_time() + command["time_to_simulate_after"])

            # Save recording
            visualizer.StopRecording()
            visualizer.PublishRecording()

            # TODO: Move this to the logger
            html = meshcat.StaticHtml()
            with open(os.path.join(self._logger._logging_path, f"{'inner' if i else 'outer'}.html"), "w") as f:
                f.write(html)

            self._logger.log_poses(context, is_outer=(i == 0))
