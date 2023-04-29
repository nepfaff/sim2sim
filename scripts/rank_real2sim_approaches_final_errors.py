"""
Script for ranking different real2sim approaches based on final errors in translation, rotation, and velocity.

NOTE: Each approach is described completely with an experiment description file. For fair comparison, the experiment
should be the same between files with some of the real2sim pipeline components being different.
"""

import os
import yaml
import argparse
import shutil
from typing import List
import time
from pathlib import Path

import wandb

from sim2sim.util.script_utils import rank_based_on_final_errors

QUATERNION_ERROR_WEIGHT = 2.0
TRANSLATION_ERROR_WEIGHT = 5.0
ANGULAR_VELOCITY_ERROR_WEIGHT = 1.0
TRANSLATIONAL_VELOCITY_ERROR_WEIGHT = 1.0


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

    current_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    experiment_description_folder_name = Path(experiment_descriptions_path).parts[-1]
    wandb.init(
        project="sim2sim_rank_real2sim_approaches_final_errors",
        name=f"{experiment_description_folder_name}_{current_time}",
        config=vars(args),
    )

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

    rank_based_on_final_errors(
        experiment_specifications,
        logging_path,
        quaternion_error_weight=QUATERNION_ERROR_WEIGHT,
        translation_error_weight=TRANSLATION_ERROR_WEIGHT,
        angular_velocity_error_weight=ANGULAR_VELOCITY_ERROR_WEIGHT,
        translational_velocity_error_weight=TRANSLATION_ERROR_WEIGHT,
        log_wandb=True,
    )

    if logging_path_is_tmp:
        print(f"Cleaning up temporary logging folder {logging_path}")
        shutil.rmtree(logging_path)


if __name__ == "__main__":
    main()
