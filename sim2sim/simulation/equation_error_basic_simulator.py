import os
import time

from typing import Dict, List

import numpy as np

from pydrake.all import (
    DiagramBuilder,
    ModelInstanceIndex,
    MultibodyPlant,
    SceneGraph,
    Simulator,
)

from sim2sim.logging import DynamicLogger
from sim2sim.simulation import BasicSimulator


class EquationErrorBasicSimulator(BasicSimulator):
    """
    A simulator that simulates the scene for `duration` seconds. It sets the inner
    manipuland state equal to the outer manipuland state every K seconds.
    """

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLogger,
        is_hydroelastic: bool,
        manipuland_names: List[str],
        reset_seconds: float,
        skip_outer_visualization: bool = False,
    ):
        """
        :param reset_seconds: The inner manipuland state is set equal to the outer
            manipuland state every `reset_seconds` seconds.
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

        self._manipuland_names = manipuland_names
        self._reset_seconds = reset_seconds

    def simulate(self, duration: float) -> None:
        # For storing the outer manipuland pose every 'reset_seconds' seconds
        outer_manipuland_state_dicts: List[Dict[ModelInstanceIndex, np.ndarray]] = []

        for i, (diagram, visualizer, meshcat) in enumerate(
            zip(
                [self._outer_diagram, self._inner_diagram],
                [self._outer_visualizer, self._inner_visualizer],
                [self._outer_meshcat, self._inner_meshcat],
            )
        ):
            simulator = Simulator(diagram)
            simulator.Initialize()

            if i == 1 or not self._skip_outer_visualization:
                # TODO: Move `StartRecording` and `StopRecording` into logger using
                # `with` statement
                visualizer.StartRecording()

            context = simulator.get_mutable_context()
            plant: MultibodyPlant = diagram.GetSubsystemByName("plant")
            plant_context = plant.GetMyMutableContextFromRoot(context)
            manipuland_instances = [
                plant.GetModelInstanceByName(name) for name in self._manipuland_names
            ]

            start_time = time.time()

            for j, t in enumerate(np.arange(0.0, duration, self._reset_seconds)):
                if i == 0:  # Outer
                    state_dict: Dict[ModelInstanceIndex, np.ndarray] = {}
                    for instance in manipuland_instances:
                        state = plant.GetPositionsAndVelocities(plant_context, instance)
                        state_dict[instance] = state
                    outer_manipuland_state_dicts.append(state_dict)
                else:  # Inner
                    for instance, state in outer_manipuland_state_dicts[j].items():
                        plant.SetPositionsAndVelocities(
                            plant_context,
                            instance,
                            state,
                        )
                simulator.AdvanceTo(t)

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
                        self._logger._logging_path, f"{'inner' if i else 'outer'}.html"
                    ),
                    "w",
                ) as f:
                    f.write(html)

            context = simulator.get_mutable_context()
            self._logger.log_manipuland_poses(context, is_outer=(i == 0))
            self._logger.log_manipuland_contact_forces(context, is_outer=(i == 0))
