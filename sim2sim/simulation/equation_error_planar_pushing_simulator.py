from typing import List, Optional

import numpy as np

from pydrake.all import DiagramBuilder, SceneGraph

from sim2sim.logging import PlanarPushingLogger
from sim2sim.simulation import PlanarPushingSimulator


class EquationErrorPlanarPushingSimulator(PlanarPushingSimulator):
    """
    A simulator that uses a fully actuated pusher geometry to push a manipuland. It
    sets the inner manipuland pose equal to the outer manipuland pose every K seconds.
    """

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
        reset_seconds: float,
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
        :param reset_seconds: The inner manipuland pose is set equal to the outer
            manipuland pose every `reset_seconds` seconds. NOTE: This must be an integer
            multiple of `controll_period`.
        :param closed_loop_control: Whether to update the control actions based on the
            actual pusher_geometry position.
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
            manipuland_names,
            target_manipuland_name,
            controll_period,
            closed_loop_control,
            num_meters_to_move_in_manpuland_direction,
            skip_outer_visualization,
        )

        self._is_equation_error = True
        self._reset_seconds = reset_seconds

        assert np.isclose(
            reset_seconds / controll_period, round(reset_seconds / controll_period)
        ), "'reset_seconds' must be an integer multiple of 'controll_period'!"
