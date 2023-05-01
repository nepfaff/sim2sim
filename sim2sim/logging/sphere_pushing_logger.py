import os

import numpy as np
from pydrake.all import DiagramBuilder, LogVectorOutput, Context, VectorLogSink

from sim2sim.logging import DynamicLogger


class SpherePushingLogger(DynamicLogger):
    """A dynamic logger that logs additional sphere pushing experiment specific data."""

    def __init__(
        self,
        logging_frequency_hz: float,
        logging_path: str,
        kProximity: bool,
        label_to_mask: int,
        manipuland_name: str,
        manipuland_base_link_name: str,
        sphere_name: str,
    ):
        """
        :param logging_frequency_hz: The frequency at which we want to log at.
        :param logging_path: The path to the directory that we want to write the log files to.
        :param kProximity: Whether to visualize kProximity or kIllustration. Visualize kProximity if true.
        :param label_to_mask: The label that we want to save binary masks for.
        :param manipuland_name: The name of the manipuland. Required for pose logging.
        :param manipuland_base_link_name: The manipuland base link name. Required for contact result force logging.
        :param sphere_name: The name of the sphere object.
        """
        super().__init__(
            logging_frequency_hz=logging_frequency_hz,
            logging_path=logging_path,
            kProximity=kProximity,
            label_to_mask=label_to_mask,
            manipuland_name=manipuland_name,
            manipuland_base_link_name=manipuland_base_link_name,
        )

        self._sphere_name = sphere_name

        self._outer_sphere_pose_logger: VectorLogSink = None
        self._outer_sphere_poses: np.ndarray = None
        self._outer_sphere_pose_times: np.ndarray = None
        self._inner_sphere_pose_logger: VectorLogSink = None
        self._inner_sphere_poses: np.ndarray = None
        self._inner_sphere_pose_times: np.ndarray = None

    def add_sphere_pose_logging(
        self, outer_builder: DiagramBuilder, inner_builder: DiagramBuilder
    ) -> None:
        self._outer_sphere_pose_logger = LogVectorOutput(
            self._outer_plant.get_state_output_port(
                self._outer_plant.GetModelInstanceByName(self._sphere_name)
            ),
            outer_builder,
            1.0 / self._logging_frequency_hz,
        )
        self._inner_sphere_pose_logger = LogVectorOutput(
            self._inner_plant.get_state_output_port(
                self._inner_plant.GetModelInstanceByName(self._sphere_name)
            ),
            inner_builder,
            1.0 / self._logging_frequency_hz,
        )

    def log_sphere_poses(self, context: Context, is_outer: bool) -> None:
        # NOTE: This really logs state which is both pose (7,) and spatial velocity (6,)

        if is_outer:
            assert self._outer_sphere_pose_logger is not None

            log = self._outer_sphere_pose_logger.FindLog(context)
            self._outer_sphere_pose_times = log.sample_times()
            self._outer_sphere_poses = log.data().T  # Shape (t, 13)
        else:
            assert self._inner_sphere_pose_logger is not None

            log = self._inner_sphere_pose_logger.FindLog(context)
            self._inner_sphere_pose_times = log.sample_times()
            self._inner_sphere_poses = log.data().T  # Shape (t, 13)

    def save_sphere_pose_logs(self) -> None:
        if self._outer_sphere_poses is not None:
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "outer_sphere_poses.txt"),
                self._outer_sphere_poses,
            )
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "outer_sphere_pose_times.txt"),
                self._outer_sphere_pose_times,
            )
        if self._inner_sphere_poses is not None:
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "inner_sphere_poses.txt"),
                self._inner_sphere_poses,
            )
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "inner_sphere_pose_times.txt"),
                self._inner_sphere_pose_times,
            )

    def save_data(self) -> None:
        self.save_sphere_pose_logs()
        super().save_data()
