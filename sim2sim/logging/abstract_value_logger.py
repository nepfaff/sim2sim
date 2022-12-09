import copy
from typing import Any, Tuple, List

from pydrake.all import LeafSystem, AbstractValue


class AbstractValueLogger(LeafSystem):
    def __init__(self, model_value: Any, logging_frequency_hz: float):
        """
        :param model_value: The abstract value to log, e.g. `model_value=ContactResults()`.
        :param logging_frequency_hz: The logging frequency.
        """
        super().__init__()

        self.DeclarePeriodicPublishEvent(period_sec=1 / logging_frequency_hz, offset_sec=0.0, publish=self.DoPublish)

        self.contact_results_input_port = self.DeclareAbstractInputPort("value", AbstractValue.Make(model_value))

        self._sample_times = []
        self._values = []

    def DoPublish(self, context, event):
        self._sample_times.append(context.get_time())
        self._values.append(copy.deepcopy(self.contact_results_input_port.Eval(context)))

    def get_logs(self) -> Tuple[List[Any], List[float]]:
        """
        :return: A tuple of (logged_values, sample_times).
        """
        return self._values, self._sample_times
