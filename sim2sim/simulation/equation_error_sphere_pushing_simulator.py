import os
import time
from typing import Optional

import numpy as np
from pydrake.all import DiagramBuilder, SceneGraph, Simulator, MultibodyPlant

from sim2sim.logging import SpherePushingLogger
from sim2sim.simulation import SpherePushingSimulator
from sim2sim.util import SphereStateSource


class EquationErrorSpherePushingSimulator(SpherePushingSimulator):
    """
    A simulator that uses a fully actuated sphere to push a manipuland. It sets the inner
    manipuland pose equal to the outer manipuland pose every K seconds.
    """

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: SpherePushingLogger,
        is_hydroelastic: bool,
        settling_time: float,
        manipuland_name: str,
        controll_period: float,
        reset_seconds: float,
        closed_loop_control: bool,
        num_meters_to_move_in_manpuland_direction: Optional[float] = None,
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
        :param manipuland_name: The name of the manipuland model instance.
        :param controll_period: Period at which to update the control command.
        :param reset_seconds: The inner manipuland pose is set equal to the outer
            manipuland pose every `reset_seconds` seconds. NOTE: This must be an integer
            multiple of `controll_period`.
        :param closed_loop_control: Whether to update the control actions based on the
            actual sphere position.
        :param num_meters_to_move_in_manpuland_direction: The number of meters to move
            the spere towards the manipuland. This is only needed/used if
            `closed_loop_control` is False.
        """
        super().__init__(
            outer_builder,
            outer_scene_graph,
            inner_builder,
            inner_scene_graph,
            logger,
            is_hydroelastic,
            settling_time,
            manipuland_name,
            controll_period,
            closed_loop_control,
            num_meters_to_move_in_manpuland_direction,
        )

        self._reset_seconds = reset_seconds

        assert np.isclose(
            reset_seconds / controll_period, round(reset_seconds / controll_period)
        ), "'reset_seconds' must be an integer multiple of 'controll_period'!"

    def simulate(self, duration: float) -> None:
        outer_manipuland_states = []
        outer_sphere_states = []

        for i, (diagram, visualizer, meshcat) in enumerate(
            zip(
                [self._outer_diagram, self._inner_diagram],
                [self._outer_visualizer, self._inner_visualizer],
                [self._outer_meshcat, self._inner_meshcat],
            )
        ):
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()
            sphere_state_source: SphereStateSource = diagram.GetSubsystemByName(
                "sphere_state_source"
            )
            plant: MultibodyPlant = diagram.GetSubsystemByName("plant")
            plant_context = plant.GetMyMutableContextFromRoot(context)
            manipuland_instance = plant.GetModelInstanceByName(self._manipuland_name)
            sphere_instance = plant.GetModelInstanceByName("sphere")

            # TODO: Move `StartRecording` and `StopRecording` into logger using `with` statement
            visualizer.StartRecording()

            start_time = time.time()

            simulator.AdvanceTo(self._settling_time)

            if i == 0:
                plant = diagram.GetSubsystemByName("plant")
                plant_context = plant.GetMyContextFromRoot(context)
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
                            plant_context, manipuland_instance
                        )[4:]

                        sphere_state_source.set_desired_position(manipuland_translation)
                        action_log.append(manipuland_translation)

                        if np.isclose(
                            sim_time / self._reset_seconds,
                            round(sim_time / self._reset_seconds),
                        ):
                            manipuland_state = plant.GetPositionsAndVelocities(
                                plant_context, manipuland_instance
                            )
                            outer_manipuland_states.append(manipuland_state)
                            sphere_state = plant.GetPositionsAndVelocities(
                                plant_context, sphere_instance
                            )
                            outer_sphere_states.append(sphere_state)

                        simulator.AdvanceTo(sim_time)
                else:
                    manipuland_translation = plant.GetPositions(
                        plant_context, manipuland_instance
                    )[4:]
                    sphere_instance = plant.GetModelInstanceByName("sphere")
                    sphere_translation = plant.GetPositions(
                        plant_context, sphere_instance
                    )

                    # Move sphere along vector connecting sphere starting position and manipuland position
                    push_direction = manipuland_translation - sphere_translation
                    push_direction_unit = push_direction / np.linalg.norm(
                        push_direction
                    )
                    action_log = (
                        sphere_translation
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
                        sphere_state_source.set_desired_position(action)

                        if np.isclose(
                            sim_time / self._reset_seconds,
                            round(sim_time / self._reset_seconds),
                        ):
                            manipuland_state = plant.GetPositionsAndVelocities(
                                plant_context, manipuland_instance
                            )
                            outer_manipuland_states.append(manipuland_state)
                            sphere_state = plant.GetPositionsAndVelocities(
                                plant_context, sphere_instance
                            )
                            outer_sphere_states.append(sphere_state)

                        simulator.AdvanceTo(sim_time)
            else:
                idx = 0
                for sim_time, action in zip(sim_times, action_log):
                    sphere_state_source.set_desired_position(action)

                    if np.isclose(
                        sim_time / self._reset_seconds,
                        round(sim_time / self._reset_seconds),
                    ):
                        plant.SetPositionsAndVelocities(
                            plant_context,
                            manipuland_instance,
                            outer_manipuland_states[idx],
                        )
                        plant.SetPositionsAndVelocities(
                            plant_context,
                            sphere_instance,
                            outer_sphere_states[idx],
                        )
                        idx += 1

                    simulator.AdvanceTo(sim_time)

            time_taken_to_simulate = time.time() - start_time
            if i == 0:
                self._logger.log(outer_simulation_time=time_taken_to_simulate)
            else:
                self._logger.log(inner_simulation_time=time_taken_to_simulate)

            visualizer.StopRecording()
            visualizer.PublishRecording()

            # TODO: Move this to the logger
            html = meshcat.StaticHtml()
            with open(
                os.path.join(
                    self._logger._logging_path, f"{'outer' if i == 0 else 'inner'}.html"
                ),
                "w",
            ) as f:
                f.write(html)

            context = simulator.get_mutable_context()
            self._logger.log_manipuland_poses(context, is_outer=(i == 0))
            self._logger.log_manipuland_contact_forces(context, is_outer=(i == 0))
            self._logger.log_sphere_poses(context, is_outer=(i == 0))
