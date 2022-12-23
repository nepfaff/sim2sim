from abc import ABC, abstractmethod

from pydrake.all import DiagramBuilder, SceneGraph

from sim2sim.logging.dynamic_logger_base import DynamicLoggerBase


class SimulatorBase(ABC):
    """Simulator base class."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLoggerBase,
    ):
        """
        :param outer_builder: Diagram builder for the outer simulation environment.
        :param outer_scene_graph: Scene graph for the outer simulation environment.
        :param inner_builder: Diagram builder for the inner simulation environment.
        :param inner_scene_graph: Scene graph for the inner simulation environment.
        :param logger: The logger.
        """
        self._outer_builder = outer_builder
        self._outer_scene_graph = outer_scene_graph
        self._inner_builder = inner_builder
        self._inner_scene_graph = inner_scene_graph
        self._logger = logger

    @abstractmethod
    def simulate(self, duration: float) -> None:
        """Simulate both the outer and inner simulation environment for `duration` seconds."""
        raise NotImplementedError
