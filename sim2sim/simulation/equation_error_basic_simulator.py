import os
import time

import numpy as np
from pydrake.all import (
    DiagramBuilder,
    SceneGraph,
    Simulator,
    MultibodyPlant,
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
        manipuland_name: str,
        reset_seconds: float,
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
        )

        self._manipuland_name = manipuland_name
        self._reset_seconds = reset_seconds

    def simulate(self, duration: float) -> None:
        # For storing the outer manipuland pose every 'reset_seconds' seconds
        outer_manipuland_states = []

        for i, (diagram, visualizer, meshcat) in enumerate(
            zip(
                [self._outer_diagram, self._inner_diagram],
                [self._outer_visualizer, self._inner_visualizer],
                [self._outer_meshcat, self._inner_meshcat],
            )
        ):
            simulator = Simulator(diagram)
            simulator.Initialize()
            # TODO: Move `StartRecording` and `StopRecording` into logger using `with` statement
            visualizer.StartRecording()

            context = simulator.get_mutable_context()
            plant: MultibodyPlant = diagram.GetSubsystemByName("plant")
            plant_context = plant.GetMyMutableContextFromRoot(context)
            manipuland_instance = plant.GetModelInstanceByName(self._manipuland_name)

            start_time = time.time()

            for j, t in enumerate(np.arange(0.0, duration, self._reset_seconds)):
                if i == 0:  # Outer
                    state = plant.GetPositionsAndVelocities(
                        plant_context, manipuland_instance
                    )
                    outer_manipuland_states.append(state)
                else:  # Inner
                    plant.SetPositionsAndVelocities(
                        plant_context,
                        manipuland_instance,
                        outer_manipuland_states[j],
                    )
                simulator.AdvanceTo(t)

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
                    self._logger._logging_path, f"{'inner' if i else 'outer'}.html"
                ),
                "w",
            ) as f:
                f.write(html)

            context = simulator.get_mutable_context()
            self._logger.log_manipuland_poses(context, is_outer=(i == 0))
            self._logger.log_manipuland_contact_forces(context, is_outer=(i == 0))
