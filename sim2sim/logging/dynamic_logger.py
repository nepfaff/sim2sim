from typing import Tuple

from pydrake.all import (
    DiagramBuilder,
    SceneGraph,
    Meshcat,
    StartMeshcat,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    ContactVisualizerParams,
    ContactVisualizer,
    MultibodyPlant,
)

from sim2sim.logging import DynamicLoggerBase


class DynamicLogger(DynamicLoggerBase):
    def __init__(self, logging_frequency_hz: float, logging_path: str):
        super().__init__(logging_frequency_hz, logging_path)

    @staticmethod
    def _add_meshcat_visualizer(
        builder: DiagramBuilder, scene_graph: SceneGraph, kProximity: bool = False
    ) -> Tuple[MeshcatVisualizer, Meshcat]:
        """
        Adds a meshcat visualizer to `builder`.

        :param builder: The diagram builder to add the visualizer to.
        :param scene_graph: The scene graph of the scene to visualize.
        :param kProximity: Whether to visualize kProximity or kIllustration. Visualize kProximity if true.
        :return: A tuple of (visualizer, meshcat).

        NOTE: Should we move this inside the logger?
        """
        meshcat = StartMeshcat()
        meshcat_params = MeshcatVisualizerParams()
        meshcat_params.role = Role.kProximity if kProximity else Role.kIllustration
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph.get_query_output_port(), meshcat, meshcat_params
        )
        return visualizer, meshcat

    @staticmethod
    def _add_contact_visualizer(builder: DiagramBuilder, meshcat: Meshcat, plant: MultibodyPlant) -> None:
        """
        Adds a contact visualizer to `builder`.

        :param builder: The diagram builder to add the visualizer to.
        :param meshcat: The meshcat that we want to add the contact visualizer to.
        :param plant: The plant for which we want to visualize contact forces.
        """
        cparams = ContactVisualizerParams()
        cparams.force_threshold = 1e-2
        cparams.newtons_per_meter = 1e6
        cparams.newton_meters_per_meter = 1e1
        cparams.radius = 0.002
        _ = ContactVisualizer.AddToBuilder(builder, plant, meshcat, cparams)

    def add_visualizers(
        self, builder: DiagramBuilder, scene_graph: SceneGraph, is_outer: bool
    ) -> Tuple[MeshcatVisualizer, Meshcat]:
        """
        Add visualizers.
        :param builder: The diagram builder to add the visualizer to.
        :param scene_graph: The scene graph of the scene to visualize.
        :return: A tuple of (visualizer, meshcat).
        """
        visualizer, meshcat = self._add_meshcat_visualizer(builder, scene_graph)
        if self._inner_plant is not None and self._outer_plant is not None:
            self._add_contact_visualizer(builder, meshcat, self._outer_plant if is_outer else self._inner_plant)
        return visualizer, meshcat

    def log(self, **kwargs) -> None:
        """TODO"""
        raise NotImplementedError

    def postprocess_data(self) -> None:
        """TODO"""
        raise NotImplementedError
