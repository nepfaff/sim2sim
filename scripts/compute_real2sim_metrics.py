"""Script for computing error metrics between simulated and real trajectories."""

import argparse
import csv

from typing import Dict, Union

import numpy as np

from sim2sim.util import (
    average_displacement_error_translation_only,
    final_displacement_error_translation_only,
    orientation_considered_average_displacement_error,
    orientation_considered_final_displacement_error,
    trajectory_IoU,
)


def create_metric_dict(
    name: str,
    trajectory_IoU_margin_01: float,
    trajectory_IoU_margin_1: float,
    orientation_considered_final_error: float,
    orientation_considered_average_error: float,
    orientation_considered_average_error_10points: float,
    final_translation_error: float,
    average_translation_error: float,
) -> Dict[str, Union[str, float]]:
    return {
        "name": name,
        "trajectory_IoU_margin_01": trajectory_IoU_margin_01,
        "trajectory_IoU_margin_1": trajectory_IoU_margin_1,
        "orientation_considered_final_error": orientation_considered_final_error,
        "orientation_considered_average_error": orientation_considered_average_error,
        "orientation_considered_average_error_10points": orientation_considered_average_error_10points,
        "final_translation_error": final_translation_error,
        "average_translation_error": average_translation_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="The name of the experiment.",
    )
    parser.add_argument(
        "--real_poses_path",
        required=True,
        type=str,
        help="The path to the numpy file containing the real poses. The poses should be "
        + "of shape (M, 7) where M is the number of objects.",
    )
    parser.add_argument(
        "--sim_states_path",
        required=True,
        type=str,
        help="The path to the numpy file containing the sim states. The states should "
        + "be of shape (M, 13) where M is the number of objects.",
    )
    parser.add_argument(
        "--num_trajectory_iou_samples",
        default=0,
        type=int,
        help="The number of samples to use for computing trajectory IoU.",
    )
    parser.add_argument(
        "--out_path",
        required=True,
        type=str,
        help="The path to the output csv file.",
    )
    args = parser.parse_args()
    name = args.name
    real_poses_path = args.real_poses_path
    sim_states_path = args.sim_states_path
    num_trajectory_iou_samples = args.num_trajectory_iou_samples
    out_path = args.out_path

    real_poses = np.load(real_poses_path)  # Shape (M,7)
    sim_states = np.load(sim_states_path)  # Shape (M,13)

    # Set velocities of real data to NaN
    real_states = np.concatenate(
        (real_poses, np.full((len(real_poses), 6), np.nan)), axis=1
    )

    # Compute metric for each manipuland separately
    trajectory_IoU_margins_01 = [
        trajectory_IoU(
            real,
            sim,
            margin=0.1,
            num_samples=num_trajectory_iou_samples,
        )
        for real, sim in zip(real_states, sim_states)
    ]  # Shape (M,)
    trajectory_IoU_margins_1 = [
        trajectory_IoU(
            real,
            sim,
            margin=1.0,
            num_samples=num_trajectory_iou_samples,
        )
        for real, sim in zip(real_states, sim_states)
    ]  # Shape (M,)
    orientation_considered_final_errors = [
        orientation_considered_final_displacement_error(real, sim)
        for real, sim in zip(real_states, sim_states)
    ]  # Shape (M,)
    orientation_considered_average_errors = [
        orientation_considered_average_displacement_error(real, sim)
        for real, sim in zip(real_states, sim_states)
    ]  # Shape (M,)
    orientation_considered_average_errors_10points = [
        orientation_considered_average_displacement_error(real, sim, 10)
        for real, sim in zip(real_states, sim_states)
    ]  # Shape (M,)
    final_displacement_errors_translation_only = [
        final_displacement_error_translation_only(real, sim)
        for real, sim in zip(real_states, sim_states)
    ]  # Shape (M,)
    average_displacement_errors_translation_only = [
        average_displacement_error_translation_only(real, sim)
        for real, sim in zip(real_states, sim_states)
    ]  # Shape (M,)

    # Average metrics over manipulands
    metrics = create_metric_dict(
        name=name,
        trajectory_IoU_margin_01=np.mean(trajectory_IoU_margins_01),
        trajectory_IoU_margin_1=np.mean(trajectory_IoU_margins_1),
        orientation_considered_final_error=np.mean(orientation_considered_final_errors),
        orientation_considered_average_error=np.mean(
            orientation_considered_average_errors
        ),
        orientation_considered_average_error_10points=np.mean(
            orientation_considered_average_errors_10points
        ),
        final_translation_error=np.mean(final_displacement_errors_translation_only),
        average_translation_error=np.mean(average_displacement_errors_translation_only),
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        column_names = metrics.keys()
        writer.writerow(column_names)
        writer.writerow([metrics[name] for name in column_names])


if __name__ == "__main__":
    main()
