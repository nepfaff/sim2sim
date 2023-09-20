import os
import time

from typing import Dict, List, Optional, Tuple

import numpy as np

from pydrake.all import (
    Context,
    DiagramBuilder,
    ModelInstanceIndex,
    MultibodyPlant,
    SceneGraph,
    Simulator,
)

from sim2sim.logging import PlanarPushingLogger
from sim2sim.simulation import SimulatorBase
from sim2sim.util import StateSource


class PlanarPushingSimulator(SimulatorBase):
    """A simulator that uses a fully actuated primitive to push a manipuland."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: PlanarPushingLogger,
        is_hydroelastic: bool,
        settling_time: float,
        manipuland_names: List[str],
        target_manipuland_name: str,
        controll_period: float,
        closed_loop_control: bool,
        num_meters_to_move_in_manpuland_direction: Optional[float] = None,
        skip_outer_visualization: bool = False,
    ):
        """
        :param outer_builder: Diagram builder for the outer simulation environment.
        :param outer_scene_graph: Scene graph for the outer simulation environment.
        :param inner_builder: Diagram builder for the inner simulation environment.
        :param inner_scene_graph: Scene graph for the inner simulation environment.
        :param logger: The logger.
        :param is_hydroelastic: Whether hydroelastic or point contact is used.
        :param settling_time: The time in seconds to simulate initially to allow the
            scene to settle.
        :param manipuland_names: The names of the manipuland model instances.
        :param target_manipuland_name: The name of the manipuland that is the pushing
            target.
        :param controll_period: Period at which to update the control command.
        :param closed_loop_control: Whether to update the control actions based on the
            actual pusher geometry position.
        :param num_meters_to_move_in_manpuland_direction: The number of meters to move
            the pusher geometry towards the manipuland. This is only needed/used if
            `closed_loop_control` is False.
        """
        super().__init__(
            outer_builder,
            outer_scene_graph,
            inner_builder,
            inner_scene_graph,
            logger,
            is_hydroelastic,
            skip_outer_visualization,
        )

        self._settling_time = settling_time
        self._manipuland_names = manipuland_names
        self._target_manipuland_name = target_manipuland_name
        self._controll_period = controll_period
        self._closed_loop_control = closed_loop_control
        self._num_meters_to_move_in_manpuland_direction = (
            num_meters_to_move_in_manpuland_direction
        )

        # To be set by 'EquationErrorPlanarPushingSimulator'
        self._is_equation_error = False
        self._reset_seconds = np.nan

        self._finalize_and_build_diagrams()

    def _finalize_and_build_diagrams(self) -> None:
        """Adds visualization systems to the outer and inner diagrams and builds them."""
        if self._skip_outer_visualization:
            self._outer_visualizer, self._outer_meshcat = None, None
        else:
            self._outer_visualizer, self._outer_meshcat = self._logger.add_visualizers(
                self._outer_builder,
                self._outer_scene_graph,
                self._is_hydroelastic,
                is_outer=True,
            )
        self._inner_visualizer, self._inner_meshcat = self._logger.add_visualizers(
            self._inner_builder,
            self._inner_scene_graph,
            self._is_hydroelastic,
            is_outer=False,
        )

        self._logger.add_manipuland_pose_logging(
            self._outer_builder, self._inner_builder
        )
        self._logger.add_manipuland_contact_force_logging(
            self._outer_builder, self._inner_builder
        )
        self._logger.add_contact_result_logging(
            self._outer_builder, self._inner_builder
        )
        self._logger.add_pusher_geometry_pose_logging(
            self._outer_builder, self._inner_builder
        )

        self._outer_diagram = self._outer_builder.Build()
        self._inner_diagram = self._inner_builder.Build()

    def _is_reset_time(self, sim_time: float) -> bool:
        return self._is_equation_error and np.isclose(
            sim_time / self._reset_seconds,
            round(sim_time / self._reset_seconds),
        )

    def _get_states(
        self,
        plant: MultibodyPlant,
        plant_context: Context,
        manipuland_instances: List[ModelInstanceIndex],
        pusher_geometry_instance: ModelInstanceIndex,
    ) -> Tuple[Dict[ModelInstanceIndex, np.ndarray], np.ndarray]:
        """
        Returns a tuple of (manipuland state dict, pusher geometry state).
        """
        manipuland_state_dict: Dict[ModelInstanceIndex, np.ndarray] = {}
        for instance in manipuland_instances:
            state = plant.GetPositionsAndVelocities(plant_context, instance)
            manipuland_state_dict[instance] = state

        pusher_geometry_state = plant.GetPositionsAndVelocities(
            plant_context, pusher_geometry_instance
        )

        return manipuland_state_dict, pusher_geometry_state

    def simulate(self, duration: float) -> None:
        outer_manipuland_state_dicts: List[Dict[ModelInstanceIndex, np.ndarray]] = []
        outer_pusher_geometry_states: List[np.ndarray] = []

        for i, (diagram, visualizer, meshcat) in enumerate(
            zip(
                [self._outer_diagram, self._inner_diagram],
                [self._outer_visualizer, self._inner_visualizer],
                [self._outer_meshcat, self._inner_meshcat],
            )
        ):
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()
            pusher_geometry_state_source: StateSource = diagram.GetSubsystemByName(
                "pusher_geometry_state_source"
            )
            plant: MultibodyPlant = diagram.GetSubsystemByName("plant")
            plant_context = plant.GetMyMutableContextFromRoot(context)
            target_manipuland_instance = plant.GetModelInstanceByName(
                self._target_manipuland_name
            )
            manipuland_instances = [
                plant.GetModelInstanceByName(name) for name in self._manipuland_names
            ]
            pusher_geometry_instance = plant.GetModelInstanceByName("pusher_geometry")

            if i == 1 or not self._skip_outer_visualization:
                # TODO: Move `StartRecording` and `StopRecording` into logger using
                # `with` statement
                visualizer.StartRecording()

            start_time = time.time()

            simulator.AdvanceTo(self._settling_time)

            if i == 0:
                total_time = self._settling_time + duration
                sim_times = np.arange(
                    self._settling_time,
                    total_time,
                    self._controll_period,
                )

                if self._closed_loop_control:
                    action_log = []
                    for sim_time in sim_times:
                        manipuland_translation = plant.GetPositions(
                            plant_context, target_manipuland_instance
                        )[4:]

                        pusher_geometry_state_source.set_desired_position(
                            manipuland_translation
                        )
                        action_log.append(manipuland_translation)

                        if self._is_reset_time(sim_time):
                            (
                                manipuland_state_dict,
                                pusher_geometry_state,
                            ) = self._get_states(
                                plant,
                                plant_context,
                                manipuland_instances,
                                pusher_geometry_instance,
                            )
                            outer_manipuland_state_dicts.append(manipuland_state_dict)
                            outer_pusher_geometry_states.append(pusher_geometry_state)

                        simulator.AdvanceTo(sim_time)
                else:
                    manipuland_translation = plant.GetPositions(
                        plant_context, target_manipuland_instance
                    )[4:]
                    pusher_geometry_instance = plant.GetModelInstanceByName(
                        "pusher_geometry"
                    )
                    pusher_geometry_translation = plant.GetPositions(
                        plant_context, pusher_geometry_instance
                    )

                    # Move pusher_geometry along vector connecting pusher_geometry
                    # starting position and manipuland position
                    push_direction = (
                        manipuland_translation - pusher_geometry_translation
                    )
                    push_direction_unit = push_direction / np.linalg.norm(
                        push_direction
                    )
                    action_log = (
                        pusher_geometry_translation
                        + (
                            np.linspace(
                                0.0,
                                self._num_meters_to_move_in_manpuland_direction,
                                len(sim_times),
                            )
                            * push_direction_unit[:, np.newaxis]
                        ).T
                    )

                    for sim_time, action in zip(sim_times, action_log):
                        pusher_geometry_state_source.set_desired_position(action)

                        if self._is_reset_time(sim_time):
                            (
                                manipuland_state_dict,
                                pusher_geometry_state,
                            ) = self._get_states(
                                plant,
                                plant_context,
                                manipuland_instances,
                                pusher_geometry_instance,
                            )
                            outer_manipuland_state_dicts.append(manipuland_state_dict)
                            outer_pusher_geometry_states.append(pusher_geometry_state)

                        simulator.AdvanceTo(sim_time)
            else:
                idx = 0
                for sim_time, action in zip(sim_times, action_log):
                    pusher_geometry_state_source.set_desired_position(action)

                    if self._is_reset_time(sim_time):
                        for instance, state in outer_manipuland_state_dicts[
                            idx
                        ].items():
                            plant.SetPositionsAndVelocities(
                                plant_context,
                                instance,
                                state,
                            )
                        plant.SetPositionsAndVelocities(
                            plant_context,
                            pusher_geometry_instance,
                            outer_pusher_geometry_states[idx],
                        )
                        idx += 1

                    simulator.AdvanceTo(sim_time)

            time_taken_to_simulate = time.time() - start_time
            if i == 0:
                self._logger.log(outer_simulation_time=time_taken_to_simulate)
            else:
                self._logger.log(inner_simulation_time=time_taken_to_simulate)

            if i == 1 or not self._skip_outer_visualization:
                visualizer.StopRecording()
                visualizer.PublishRecording()

                # TODO: Move this to the logger
                html = meshcat.StaticHtml()
                with open(
                    os.path.join(
                        self._logger._logging_path,
                        f"{'outer' if i == 0 else 'inner'}.html",
                    ),
                    "w",
                ) as f:
                    f.write(html)

            context = simulator.get_mutable_context()
            self._logger.log_manipuland_poses(context, is_outer=(i == 0))
            self._logger.log_manipuland_contact_forces(context, is_outer=(i == 0))
            self._logger.log_pusher_geometry_poses(context, is_outer=(i == 0))
