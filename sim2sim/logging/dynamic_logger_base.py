from abc import ABC, abstractmethod


class DynamicLoggerBase(ABC):
    """TODO"""

    def __init__(self, logging_frequency_hz: float, logging_path: str):
        """TODO"""
        self._logging_frequency_hz = logging_frequency_hz
        self._logging_path = logging_path

    @abstractmethod
    def log(self, **kwargs) -> None:
        """TODO"""
        raise NotImplementedError

    @abstractmethod
    def postprocess_data(self) -> None:
        """TODO"""
        raise NotImplementedError
