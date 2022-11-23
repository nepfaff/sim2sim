from pydrake.all import (
    DiagramBuilder,
    SceneGraph,
    Simulator,
)

from sim2sim.logging import DynamicLoggerBase
from sim2sim.simulation import SimulatorBase


class TablePIDSimulator(SimulatorBase):
    """The simulator for the table PID scene."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLoggerBase,
    ):
        super().__init__(outer_builder, outer_scene_graph, inner_builder, inner_scene_graph, logger)
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

        self._outer_diagram = self._outer_builder.Build()
        self._inner_diagram = self._inner_builder.Build()

    def simulate(self, duration: float) -> None:
        # TODO: Parallelise the two simulations
        for i, (diagram, visualizer, meshcat) in enumerate(
            zip(
                [self._outer_diagram, self._inner_diagram],
                [self._outer_visualizer, self._inner_visualizer],
                [self._outer_meshcat, self._inner_meshcat],
            )
        ):
            simulator = Simulator(diagram)
            simulator.Initialize()
            visualizer.StartRecording()
            simulator.AdvanceTo(duration)
            visualizer.StopRecording()
            visualizer.PublishRecording()

            # TODO: Move this to the logger
            html = meshcat.StaticHtml()
            with open(f"test{i}", "w") as f:
                f.write(html)
