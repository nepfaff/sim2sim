"""
Script for post-processing the dataset for metric learning from the data generated by
'collect_random_force_data.py.
"""

import argparse
import os

import numpy as np
from matplotlib import pyplot as plt


def process_perturbation(perturbation_entry_path: str) -> None:
    manipuland_state_errors = []
    for run_entry in os.scandir(perturbation_entry_path):
        if run_entry.is_dir():
            times_path = os.path.join(
                run_entry.path, "time_logs", "outer_manipuland_pose_times.txt"
            )
            if not os.path.exists(times_path):
                print(f"Data incomplete for {run_entry}")
                continue
            times = np.loadtxt(times_path)
            outer_manipuland_states_path = os.path.join(
                run_entry.path, "time_logs", "outer_manipuland_poses.npy"
            )
            inner_manipuland_states_path = os.path.join(
                run_entry.path, "time_logs", "inner_manipuland_poses.npy"
            )
            outer_states = np.load(outer_manipuland_states_path)
            inner_states = np.load(inner_manipuland_states_path)
            # Mean over manipulands
            manipuland_state_error = np.mean(outer_states - inner_states, axis=0)
            manipuland_state_errors.append(manipuland_state_error)

    manipuland_orientation_error = manipuland_state_errors[:, :, :4]  # Quaternions
    manipuland_translation_error = manipuland_state_errors[:, :, 4:7]
    manipuland_angular_velocity_error = manipuland_state_errors[:, :, 7:10]
    manipuland_translational_velocity_error = manipuland_state_errors[:, :, 10:]

    np.savetxt(
        os.path.join(
            perturbation_entry_path, "mean_final_manipuland_state_error_magnitude.txt"
        ),
        [np.mean(np.linalg.norm(manipuland_state_errors[:, -1], axis=1))],
    )

    plt.plot(
        times, np.mean(np.linalg.norm(manipuland_translation_error, axis=2), axis=0)
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Mean translation error magnitude (m)")
    plt.savefig(
        os.path.join(
            perturbation_entry_path, "mean_manipuland_translation_error_magnitude.png"
        )
    )
    plt.close()

    plt.plot(
        times, np.mean(np.linalg.norm(manipuland_orientation_error, axis=2), axis=0)
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Mean orientation error magnitude (quaternions)")
    plt.savefig(
        os.path.join(
            perturbation_entry_path, "mean_manipuland_orientation_error_magnitude.png"
        )
    )
    plt.close()

    plt.plot(
        times,
        np.mean(
            np.linalg.norm(manipuland_translational_velocity_error, axis=2), axis=0
        ),
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Mean translational velocity error magnitude (m/s)")
    plt.savefig(
        os.path.join(
            perturbation_entry_path,
            "mean_manipuland_translational_velocity_error_magnitude.png",
        )
    )
    plt.close()

    plt.plot(
        times,
        np.mean(np.linalg.norm(manipuland_angular_velocity_error, axis=2), axis=0),
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Mean angular velocity error magnitude (rad/s)")
    plt.savefig(
        os.path.join(
            perturbation_entry_path,
            "mean_manipuland_angular_velocity_error_magnitude.png",
        )
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="The top level path for the data generated by 'collect_random_force_data.py'.",
    )
    parser.add_argument(
        "--is_representation_specific",
        action="store_true",
        help="Whether to the directory structure matches the one generated by "
        + "'collect_representation_specific_random_force_data.py'.",
    )
    args = parser.parse_args()

    if args.is_representation_specific:
        process_perturbation(args.data_path)
    else:
        for perturbation_entry in os.scandir(args.data_path):
            if perturbation_entry.is_dir():
                process_perturbation(perturbation_entry.path)


if __name__ == "__main__":
    main()
