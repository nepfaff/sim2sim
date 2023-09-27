import os
import time

from pydrake.all import DiagramBuilder, SceneGraph, Simulator

from sim2sim.logging import DynamicLogger

from .simulator_base import SimulatorBase


class BasicSimulator(SimulatorBase):
    """A simulator that simply simulates the scene for `duration` seconds."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLogger,
        is_hydroelastic: bool,
        skip_outer_visualization: bool = False,
    ):
        super().__init__(
            outer_builder,
            outer_scene_graph,
            inner_builder,
            inner_scene_graph,
            logger,
            is_hydroelastic,
            skip_outer_visualization,
        )
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

            if i == 1 or not self._skip_outer_visualization:
                # TODO: Move `StartRecording` and `StopRecording` into logger using
                # `with` statement
                visualizer.StartRecording()

            start_time = time.time()

            simulator.AdvanceTo(duration)

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
