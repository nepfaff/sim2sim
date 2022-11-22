from abc import ABC, abstractmethod

from pydrake.all import MultibodyPlant

from sim2sim.logging import DynamicLoggerBase


class SimulatorBase(ABC):
    """TODO"""

    def __init__(self, outer_plant: MultibodyPlant, inner_plant: MultibodyPlant, logger: DynamicLoggerBase):
        """

        :param outer_plant: Finalized plant of the outer simulation environment.
        :param inner_plant: Finalized plant of the inner simulation environment.
        """
        self._outer_plant = outer_plant
        self._inner_plant = inner_plant
        self._logger = logger

    @abstractmethod
    def simulate(self, duration: float):
        """TODO"""
        raise NotImplementedError
