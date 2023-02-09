import os
import time

from pydrake.all import (
    DiagramBuilder,
    SceneGraph,
    Simulator,
)

from sim2sim.logging import DynamicLogger
from sim2sim.simulation import SimulatorBase


class BasicInnerOnlySimulator(SimulatorBase):
    """A simulator that simply simulates the scene for `duration` seconds. It only simulates the inner diagram."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLogger,
        is_hydroelastic: bool,
    ):
        super().__init__(outer_builder, outer_scene_graph, inner_builder, inner_scene_graph, logger, is_hydroelastic)
        self._finalize_and_build_diagram()

    def _finalize_and_build_diagram(self) -> None:
        """Adds visualization systems to the outer and inner diagrams and builds them."""

        self._inner_visualizer, self._inner_meshcat = self._logger.add_visualizers(
            self._inner_builder,
            self._inner_scene_graph,
            self._is_hydroelastic,
            is_outer=False,
        )

        self._logger.add_manipuland_pose_logging(self._outer_builder, self._inner_builder)
        self._logger.add_manipuland_contact_force_logging(self._outer_builder, self._inner_builder)
        self._logger.add_contact_result_logging(self._outer_builder, self._inner_builder)

        self._inner_diagram = self._inner_builder.Build()

    def simulate(self, duration: float) -> None:
        simulator = Simulator(self._inner_diagram)
        simulator.Initialize()
        # TODO: Move `StartRecording` and `StopRecording` into logger using `with` statement
        self._inner_visualizer.StartRecording()

        start_time = time.time()

        simulator.AdvanceTo(duration)

        time_taken_to_simulate = time.time() - start_time
        self._logger.log(inner_simulation_time=time_taken_to_simulate)

        self._inner_visualizer.StopRecording()
        self._inner_visualizer.PublishRecording()

        # TODO: Move this to the logger
        html = self._inner_meshcat.StaticHtml()
        with open(os.path.join(self._logger._logging_path, f"inner.html"), "w") as f:
            f.write(html)

        context = simulator.get_mutable_context()
        self._logger.log_manipuland_poses(context, is_outer=False)
        self._logger.log_manipuland_contact_forces(context, is_outer=False)
