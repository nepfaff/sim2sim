import numpy as np


def _ade(outer_traj: np.ndarray, inner_traj: np.ndarray) -> float:
    error = outer_traj - inner_traj
    ade = np.mean(np.linalg.norm(error[:, :7], axis=1))
    return ade


def average_displacement_error(outer_state_trajectory: np.ndarray, inner_state_trajectory: np.ndarray) -> float:
    """
    Average Displacement Error (ADE) metric that considers final translation and orientation (velocities not
    considered).

    :param outer_state_trajectory: The trajectory of outer manipuland states of shape (N,13) where N is the number of
        trajectory points. Each point has the form [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, tz], where q are
        quaternions, t are translations, w are angular velocities, and v are translational velocities.
    :param inner_state_trajectory: The trajectory of the inner manipuland states of same shape as
        `outer_state_trajectory`.
    """
    return _ade(outer_state_trajectory[:, :7], inner_state_trajectory[:, :7])


def average_displacement_error_translation_only(
    outer_state_trajectory: np.ndarray, inner_state_trajectory: np.ndarray
) -> float:
    """
    Average Displacement Error (ADE) metric that only considers final translation (orientation and velocities not
    considered).

    :param outer_state_trajectory: The trajectory of outer manipuland states of shape (N,13) where N is the number of
        trajectory points. Each point has the form [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, tz], where q are
        quaternions, t are translations, w are angular velocities, and v are translational velocities.
    :param inner_state_trajectory: The trajectory of the inner manipuland states of same shape as
        `outer_state_trajectory`.
    """
    return _ade(outer_state_trajectory[:, 4:7], inner_state_trajectory[:, 4:7])


def _fde(outer_traj: np.ndarray, inner_traj: np.ndarray) -> float:
    error = outer_traj - inner_traj
    fde = np.linalg.norm(error[-1])
    return fde


def final_displacement_error(outer_state_trajectory: np.ndarray, inner_state_trajectory: np.ndarray) -> float:
    """
    Final Displacement Error (FDE) metric that considers final translation and orientation (velocities not
    considered).

    :param outer_state_trajectory: The trajectory of outer manipuland states of shape (N,13) where N is the number of
        trajectory points. Each point has the form [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, tz], where q are
        quaternions, t are translations, w are angular velocities, and v are translational velocities.
    :param inner_state_trajectory: The trajectory of the inner manipuland states of same shape as
        `outer_state_trajectory`.
    """
    return _fde(outer_state_trajectory[:, :7], inner_state_trajectory[:, :7])


def final_displacement_error_translation_only(
    outer_state_trajectory: np.ndarray, inner_state_trajectory: np.ndarray
) -> float:
    """
    Final Displacement Error (FDE) metric that only considers final translation (orientation and velocities not
    considered).

    :param outer_state_trajectory: The trajectory of outer manipuland states of shape (N,13) where N is the number of
        trajectory points. Each point has the form [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, tz], where q are
        quaternions, t are translations, w are angular velocities, and v are translational velocities.
    :param inner_state_trajectory: The trajectory of the inner manipuland states of same shape as
        `outer_state_trajectory`.
    """
    return _fde(outer_state_trajectory[:, 4:7], inner_state_trajectory[:, 4:7])
