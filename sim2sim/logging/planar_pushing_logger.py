import os

from typing import List

import numpy as np

from pydrake.all import Context, DiagramBuilder, LogVectorOutput, VectorLogSink

from sim2sim.logging import DynamicLogger


class PlanarPushingLogger(DynamicLogger):
    """A dynamic logger that logs additional planar pushing experiment specific data."""

    def __init__(
        self,
        logging_frequency_hz: float,
        logging_path: str,
        kProximity: bool,
        label_to_mask: int,
        manipuland_base_link_names: List[str],
        pusher_geometry_name: str = "pusher_geometry",
    ):
        """
        :param logging_frequency_hz: The frequency at which we want to log at.
        :param logging_path: The path to the directory that we want to write the log
            files to.
        :param kProximity: Whether to visualize kProximity or kIllustration. Visualize
            kProximity if true.
        :param label_to_mask: The label that we want to save binary masks for.
        :param manipuland_base_link_names: The manipuland base link names.
        :param pusher_geometry_name: The name of the pusher_geometry object.
        """
        super().__init__(
            logging_frequency_hz=logging_frequency_hz,
            logging_path=logging_path,
            kProximity=kProximity,
            label_to_mask=label_to_mask,
            manipuland_base_link_names=manipuland_base_link_names,
        )

        self._pusher_geometry_name = pusher_geometry_name

        self._outer_pusher_geometry_pose_logger: VectorLogSink = None
        self._outer_pusher_geometry_poses: np.ndarray = None
        self._outer_pusher_geometry_pose_times: np.ndarray = None
        self._inner_pusher_geometry_pose_logger: VectorLogSink = None
        self._inner_pusher_geometry_poses: np.ndarray = None
        self._inner_pusher_geometry_pose_times: np.ndarray = None

    def add_pusher_geometry_pose_logging(
        self, outer_builder: DiagramBuilder, inner_builder: DiagramBuilder
    ) -> None:
        self._outer_pusher_geometry_pose_logger = LogVectorOutput(
            self._outer_plant.get_state_output_port(
                self._outer_plant.GetModelInstanceByName(self._pusher_geometry_name)
            ),
            outer_builder,
            1.0 / self._logging_frequency_hz,
        )
        self._inner_pusher_geometry_pose_logger = LogVectorOutput(
            self._inner_plant.get_state_output_port(
                self._inner_plant.GetModelInstanceByName(self._pusher_geometry_name)
            ),
            inner_builder,
            1.0 / self._logging_frequency_hz,
        )

    def log_pusher_geometry_poses(self, context: Context, is_outer: bool) -> None:
        # NOTE: This really logs state which is both pose (7,) and spatial velocity (6,)

        if is_outer:
            assert self._outer_pusher_geometry_pose_logger is not None

            log = self._outer_pusher_geometry_pose_logger.FindLog(context)
            self._outer_pusher_geometry_pose_times = log.sample_times()
            self._outer_pusher_geometry_poses = log.data().T  # Shape (t, 13)
        else:
            assert self._inner_pusher_geometry_pose_logger is not None

            log = self._inner_pusher_geometry_pose_logger.FindLog(context)
            self._inner_pusher_geometry_pose_times = log.sample_times()
            self._inner_pusher_geometry_poses = log.data().T  # Shape (t, 13)

    def save_pusher_geometry_pose_logs(self) -> None:
        if self._outer_pusher_geometry_poses is not None:
            np.savetxt(
                os.path.join(
                    self._time_logs_dir_path, "outer_pusher_geometry_poses.txt"
                ),
                self._outer_pusher_geometry_poses,
            )
            np.savetxt(
                os.path.join(
                    self._time_logs_dir_path, "outer_pusher_geometry_pose_times.txt"
                ),
                self._outer_pusher_geometry_pose_times,
            )
        if self._inner_pusher_geometry_poses is not None:
            np.savetxt(
                os.path.join(
                    self._time_logs_dir_path, "inner_pusher_geometry_poses.txt"
                ),
                self._inner_pusher_geometry_poses,
            )
            np.savetxt(
                os.path.join(
                    self._time_logs_dir_path, "inner_pusher_geometry_pose_times.txt"
                ),
                self._inner_pusher_geometry_pose_times,
            )

    def save_data(self) -> None:
        self.save_pusher_geometry_pose_logs()
        super().save_data()
