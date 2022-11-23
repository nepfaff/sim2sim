from abc import ABC, abstractmethod
from typing import Union
import os

from pydrake.all import MultibodyPlant


class DynamicLoggerBase(ABC):
    """Dynamics logger base class."""

    def __init__(self, logging_frequency_hz: float, logging_path: str):
        """
        :param logging_frequency_hz: The frequency at which we want to log at.
        :param logging_path: The path to the directory that we want to write the log files to.
        """
        self._logging_frequency_hz = logging_frequency_hz
        self._logging_path = logging_path

        self._outer_plant: Union[MultibodyPlant, None] = None
        self._inner_plant: Union[MultibodyPlant, None] = None

        if not os.path.exists(logging_path):
            os.mkdir(logging_path)

    def add_plants(self, outer_plant: MultibodyPlant, inner_plant: MultibodyPlant) -> None:
        """Add finalized plants."""
        self._outer_plant = outer_plant
        self._inner_plant = inner_plant

    @abstractmethod
    def log(self, **kwargs) -> None:
        """TODO"""
        raise NotImplementedError

    @abstractmethod
    def postprocess_data(self) -> None:
        """TODO"""
        raise NotImplementedError
