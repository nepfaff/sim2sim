import os
import time

import numpy as np
from pydrake.all import DiagramBuilder, SceneGraph, Simulator

from sim2sim.logging import DynamicLoggerBase
from sim2sim.simulation import SimulatorBase
from sim2sim.util import SphereStateSource


class SpherePushingSimulator(SimulatorBase):
    """A simulator that uses a fully actuated sphere to push a manipuland."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLoggerBase,
        settling_time: float,
        manipuland_name: str,
        controll_period: float,
    ):
        """
        :param outer_builder: Diagram builder for the outer simulation environment.
        :param outer_scene_graph: Scene graph for the outer simulation environment.
        :param inner_builder: Diagram builder for the inner simulation environment.
        :param inner_scene_graph: Scene graph for the inner simulation environment.
        :param logger: The logger.
        :param settling_time: The time in seconds to simulate initially to allow the scene to settle.
        :param manipuland_name: The name of the manipuland model instance.
        :param controll_period: Period at which to update the control command.
        """
        super().__init__(outer_builder, outer_scene_graph, inner_builder, inner_scene_graph, logger)

        self._settling_time = settling_time
        self._manipuland_name = manipuland_name
        self._controll_period = controll_period

        self._finalize_and_build_diagrams()

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

        self._logger.add_manipuland_pose_logging(self._outer_builder, self._inner_builder)
        self._logger.add_manipuland_contact_force_logging(self._outer_builder, self._inner_builder)
        self._logger.add_contact_result_logging(self._outer_builder, self._inner_builder)

        self._outer_diagram = self._outer_builder.Build()
        self._inner_diagram = self._inner_builder.Build()

    def simulate(self, duration: float) -> None:
        for i, (diagram, visualizer, meshcat) in enumerate(
            zip(
                [self._outer_diagram, self._inner_diagram],
                [self._outer_visualizer, self._inner_visualizer],
                [self._outer_meshcat, self._inner_meshcat],
            )
        ):
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()
            sphere_state_source: SphereStateSource = diagram.GetSubsystemByName("sphere_state_source")

            # TODO: Move `StartRecording` and `StopRecording` into logger using `with` statement
            visualizer.StartRecording()

            start_time = time.time()

            simulator.AdvanceTo(self._settling_time)

            # TODO: Save the actions from the outer sim and replay for the inner sim
            plant = diagram.GetSubsystemByName("plant")
            plant_context = plant.GetMyContextFromRoot(context)
            total_time = self._settling_time + duration
            for sim_time in np.linspace(self._settling_time, total_time, int(duration / self._controll_period)):
                mesh_manipuland_instance = plant.GetModelInstanceByName(self._manipuland_name)
                mesh_manipuland_translation = plant.GetPositions(plant_context, mesh_manipuland_instance)[4:]

                sphere_state_source.set_desired_position(mesh_manipuland_translation)

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
            with open(os.path.join(self._logger._logging_path, f"{'outer' if i == 0 else 'inner'}.html"), "w") as f:
                f.write(html)

            context = simulator.get_mutable_context()
            self._logger.log_manipuland_poses(context, is_outer=(i == 0))
            self._logger.log_manipuland_contact_forces(context, is_outer=(i == 0))
