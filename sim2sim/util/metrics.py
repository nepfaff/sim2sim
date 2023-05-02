import numpy as np
from scipy.spatial.transform import Rotation


def _ade(outer_traj: np.ndarray, inner_traj: np.ndarray) -> float:
    error = outer_traj - inner_traj
    ade = np.mean(np.linalg.norm(error[:, :7], axis=1))
    return ade


def average_displacement_error(
    outer_state_trajectory: np.ndarray, inner_state_trajectory: np.ndarray
) -> float:
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
    error = outer_traj[-1] - inner_traj[-1]
    fde = np.linalg.norm(error)
    return fde


def final_displacement_error(
    outer_state_trajectory: np.ndarray, inner_state_trajectory: np.ndarray
) -> float:
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


def trajectory_IoU(
    outer_state_trajectory: np.ndarray,
    inner_state_trajectory: np.ndarray,
    margin: float,
    num_samples: int,
) -> float:
    """
    Trajectory Intersection over Unions (TIOU) metric that considers final translation (orientations and velocities not
    considered). It is computed using a Monte Carlo approach.
    See https://towardsdatascience.com/why-ade-and-fde-might-not-be-the-best-metrics-to-score-motion-prediction-model-performance-and-what-1980366d37be.

    TODO: Look into downsampling trajectory points to speed up computation

    :param outer_state_trajectory: The trajectory of outer manipuland states of shape (N,13) where N is the number of
        trajectory points. Each point has the form [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, tz], where q are
        quaternions, t are translations, w are angular velocities, and v are translational velocities.
    :param inner_state_trajectory: The trajectory of the inner manipuland states of same shape as
        `outer_state_trajectory`.
    :param margin: The margin to use for each dimension.
    :param num_samples: The number of samples to take for each trajectory point. Total number of samples =
        num_samples * N.
    """

    def within_margin(traj_point: np.ndarray, sample: np.ndarray) -> bool:
        def within_1d(idx: int) -> bool:
            return (
                sample[idx] < traj_point[idx] + margin
                and sample[idx] > traj_point[idx] - margin
            )

        return np.all([within_1d(i) for i in range(len(traj_point))])

    intersection = 0
    union = 0
    for outer_point, inner_point in zip(
        outer_state_trajectory[:, 4:7], inner_state_trajectory[:, 4:7]
    ):
        aabb = np.min([outer_point - margin, inner_point - margin], axis=0), np.max(
            [outer_point + margin, inner_point + margin], axis=0
        )  # [minx, miny, minz], [maxx, maxy, maxz]
        aabb = np.concatenate(aabb)

        for _ in range(num_samples):
            # Take sample within aabb
            sample = aabb[:3] + np.random.rand(3) * (aabb[3:] - aabb[:3])

            # Check whether within margin of inner and/or outer trajectory point
            within_outer = within_margin(outer_point, sample)
            within_inner = within_margin(inner_point, sample)
            if within_outer or within_inner:
                union += 1

                if within_outer and within_inner:
                    intersection += 1

    iou = intersection / union
    return iou


def _states_to_poses(states: np.ndarray) -> np.ndarray:
    """
    :param states: States of shape (N,13) where N is the number of trajectory points.
        Each point has the form [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, tz],
        where q are quaternions, t are translations, w are angular velocities, and v are
        translational velocities.
    :return: Homogenous transformation matrices of shape (N,4,4).
    """
    poses = np.eye(4)[np.newaxis, :].repeat(len(states), axis=0)
    # Drake used (qw, qx, qy, qz) and scipy uses (qx, qy, qz, qw)
    poses[:, :3, :3] = Rotation.from_quat(
        np.concatenate((states[:, 1:4], states[:, :1]), axis=-1)
    ).as_matrix()
    poses[:, :3, 3] = states[:, 4:7]
    return poses


def orientation_considered_final_displacement_error(
    outer_state_trajectory: np.ndarray, inner_state_trajectory: np.ndarray
) -> float:
    """
    Final Displacement Error (FDE) metric that consideres orientation by sampling points
    relative to the object pose and taking the mean displacement error of these points.

    :param outer_state_trajectory: The trajectory of outer manipuland states of shape
        (N,13) where N is the number of trajectory points. Each point has the form
        [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, tz], where q are quaternions,
        t are translations, w are angular velocities, and v are translational velocities.
    :param inner_state_trajectory: The trajectory of the inner manipuland states of same
        shape as `outer_state_trajectory`.
    """
    final_state_outer = outer_state_trajectory[-1][np.newaxis, :]
    final_state_inner = inner_state_trajectory[-1][np.newaxis, :]

    final_pose_outer = _states_to_poses(final_state_outer).squeeze(0)
    final_pose_inner = _states_to_poses(final_state_inner).squeeze(0)

    # Sample 3 points in object frame to completely define the orientation
    points_object_frame = np.eye(3)
    points_outer_world_frame = (
        points_object_frame @ final_pose_outer[:3, :3].T + final_pose_outer[:3, 3]
    )
    points_inner_world_frame = (
        points_object_frame @ final_pose_inner[:3, :3].T + final_pose_inner[:3, 3]
    )

    pointwise_error = np.linalg.norm(
        points_outer_world_frame - points_inner_world_frame, axis=-1
    )
    mean_error = np.mean(pointwise_error)
    return mean_error


def orientation_considered_average_displacement_error(
    outer_state_trajectory: np.ndarray, inner_state_trajectory: np.ndarray
) -> float:
    """
    Average Displacement Error (ADE) metric that consideres orientation by sampling points
    relative to the object pose and taking the mean average displacement error of these
    points.

    :param outer_state_trajectory: The trajectory of outer manipuland states of shape
        (N,13) where N is the number of trajectory points. Each point has the form
        [q1, q2, q3, q4, tx, ty, tz, wx, wy, wz, vx, vy, tz], where q are quaternions,
        t are translations, w are angular velocities, and v are translational velocities.
    :param inner_state_trajectory: The trajectory of the inner manipuland states of same
        shape as `outer_state_trajectory`.
    """
    poses_outer = _states_to_poses(outer_state_trajectory)  # Shape (N,4,4)
    poses_inner = _states_to_poses(inner_state_trajectory)  # Shape (N,4,4)

    # Sample 3 points in object frame to completely define the orientation
    points_object_frame = np.eye(3)
    points_outer_world_frame = (
        points_object_frame @ poses_outer[:, :3, :3].transpose((0, 2, 1))
        + poses_outer[:, :3, 3][:, np.newaxis, :]
    )  # Shape (N,3,3)
    points_inner_world_frame = (
        points_object_frame @ poses_inner[:, :3, :3].transpose((0, 2, 1))
        + poses_inner[:, :3, 3][:, np.newaxis, :]
    )  # Shape (N,3,3)

    pointwise_error = np.linalg.norm(
        points_outer_world_frame - points_inner_world_frame, axis=-1
    )  # Shape (N,3)
    mean_error = np.mean(pointwise_error, axis=-1)  # Shape (N,)
    ade = np.mean(mean_error)
    return ade
