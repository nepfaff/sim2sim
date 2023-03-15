"""
Script for ranking different real2sim approaches based on final errors in translation, rotation, and velocity.

NOTE: Each approach is described completely with an experiment description file. For fair comparison, the experiment
should be the same between files with some of the real2sim pipeline components being different.
"""

import os
import yaml
import argparse
import glob
import shutil
from typing import List, Dict

import numpy as np
from prettytable import PrettyTable

from sim2sim.experiments import (
    run_table_pid,
    run_iiwa_manip_tomato_soup_can,
    run_iiwa_manip_pepper,
    run_floor_drop,
    run_random_force,
    run_sphere_pushing,
)

QUATERNION_ERROR_WEIGHT = 2.0
TRANSLATION_ERROR_WEIGHT = 5.0
ANGULAR_VELOCITY_ERROR_WEIGHT = 1.0
TRANSLATIONAL_VELOCITY_ERROR_WEIGHT = 1.0


ENTRYPOINTS = {
    "table_pid": run_table_pid,
    "iiwa_manip_tomato_soup_can": run_iiwa_manip_tomato_soup_can,
    "iiwa_manip_pepper": run_iiwa_manip_pepper,
    "floor_drop": run_floor_drop,
    "random_force": run_random_force,
    "sphere_pushing": run_sphere_pushing,
}


def create_evaluation_results_table(eval_data: List[Dict[str, float]]) -> PrettyTable:
    eval_data.sort(key=lambda x: x["combined_error"])

    table = PrettyTable(
        [
            "Mesh name",
            "Combined Err",
            "Translation Err",
            "Quaternion Err",
            "Translational Velocity Err",
            "Angular Velocity Err",
        ]
    )
    for el in eval_data:
        table.add_row(
            [
                el["name"],
                el["combined_error"],
                el["translation_error"],
                el["quaternion_error"],
                el["translational_velocity_error"],
                el["angular_velocity_error"],
            ]
        )

    return table


def rank_static_stability(experiment_specifications: List[dict], logging_dir_path: str) -> None:
    # TODO: Add options to parallelize this
    eval_data: List[Dict[str, float]] = []
    for experiment_specification in experiment_specifications:
        name = experiment_specification["experiment_id"]
        print(f"\nEvaluating {name}:")

        logging_path = os.path.join(logging_dir_path, name)
        if not os.path.exists(logging_path):
            os.mkdir(logging_path)

        runner = ENTRYPOINTS[experiment_specification["script"]["name"]]
        runner(
            logging_path=logging_path,
            params=experiment_specification,
            **experiment_specification["script"]["args"],
        )

        time_logs_path = os.path.join(logging_path, "time_logs")
        outer_states = np.loadtxt(os.path.join(time_logs_path, "outer_manipuland_poses.txt"))
        inner_states = np.loadtxt(os.path.join(time_logs_path, "inner_manipuland_poses.txt"))
        final_state_error = outer_states[-1] - inner_states[-1]
        quaternion_error = np.linalg.norm(final_state_error[:4])
        translation_error = np.linalg.norm(final_state_error[4:7])
        angular_velocity_error = np.linalg.norm(final_state_error[7:10])
        translational_velocity_error = np.linalg.norm(final_state_error[10:])

        eval_data.append(
            {
                "name": name,
                "combined_error": QUATERNION_ERROR_WEIGHT * quaternion_error
                + TRANSLATION_ERROR_WEIGHT * translation_error
                + ANGULAR_VELOCITY_ERROR_WEIGHT * angular_velocity_error
                + TRANSLATIONAL_VELOCITY_ERROR_WEIGHT * translational_velocity_error,
                "quaternion_error": quaternion_error,
                "translation_error": translation_error,
                "angular_velocity_error": angular_velocity_error,
                "translational_velocity_error": translational_velocity_error,
            }
        )

        print(
            f"Mesh: {name}, Translation err: {translation_error}, Quaternion err: {quaternion_error}, "
            + f"Translational velocity err: {translational_velocity_error}, "
            + f"Angular velocity err: {angular_velocity_error}"
        )

    table = create_evaluation_results_table(eval_data)
    print("\n\nEvaluation results:\n")
    print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_descriptions",
        required=True,
        type=str,
        help="The path to the folder containing the experiment descriptions for the different experiments.",
    )
    parser.add_argument(
        "--logging_path",
        required=False,
        type=str,
        help="The path to log the results to. A temporary folder will be created and deleted if not given.",
    )
    args = parser.parse_args()
    experiment_descriptions_path = args.experiment_descriptions
    logging_path = args.logging_path

    logging_path_is_tmp = False
    if not logging_path:
        logging_path = os.path.join(experiment_descriptions_path, "tmp_logs")
        logging_path_is_tmp = True

    if not os.path.exists(logging_path):
        print(f"Creating folder {logging_path}")
        os.mkdir(logging_path)

    experiment_specifications: List[dict] = []
    with os.scandir(experiment_descriptions_path) as paths:
        for path in paths:
            if path.is_file():
                experiment_description = yaml.safe_load(open(path, "r"))
                experiment_specifications.append(experiment_description)

    rank_static_stability(experiment_specifications, logging_path)

    if logging_path_is_tmp:
        print(f"Cleaning up temporary logging folder {logging_path}")
        shutil.rmtree(logging_path)


if __name__ == "__main__":
    main()
