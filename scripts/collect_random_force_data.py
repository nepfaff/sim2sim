"""Script for collecting random force data for metric learning."""

import os
import yaml
import argparse
import glob
import re

import numpy as np

from sim2sim.experiments import run_random_force

# TODO: Implement multicore execution

PERTURBATION_DIR_BASENAME = "perturb_"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_description",
        required=True,
        type=str,
        help="The path to the yaml file that descripes the experiment to run.",
    )
    parser.add_argument(
        "--logging_path",
        required=True,
        type=str,
        help="The path to log the results to.",
    )
    parser.add_argument(
        "--num_runs_per_perturbation",
        default=10,
        type=int,
        help="How many random force experiments to run per mesh perturbation.",
    )
    parser.add_argument(
        "--num_perturbations", default=1000, type=int, help="The number of different perturbations to run."
    )
    parser.add_argument("--num_cores", default=1, type=int, help="The number of cpu cores to use.")
    args = parser.parse_args()

    experiment_specification = yaml.safe_load(open(args.experiment_description, "r"))

    # Find start perturbation number to prevent overwritting already generated data
    perturb_num = 0
    if os.path.exists(args.logging_path):
        for path in glob.iglob(f"{args.logging_path}/{PERTURBATION_DIR_BASENAME}*"):
            num = int(re.findall(r"\d+", path)[-1])
            if num > perturb_num:
                perturb_num = num
    else:
        os.mkdir(args.logging_path)

    for _ in range(args.num_perturbations):
        perturb_num += 1

        perturb_path = os.path.join(args.logging_path, f"{PERTURBATION_DIR_BASENAME}{perturb_num:06d}")
        os.mkdir(perturb_path)

        # Random perturbation
        experiment_specification["mesh_processor"]["args"]["gmm_em_params"] = {
            "n_components": int(np.random.uniform(1, 150)),
            "tol": np.random.uniform(0.0, 0.01),
            "max_iter": int(np.random.normal(100, 15)),
            "n_init": 1 + int(np.random.choice(5)),
            "init_params": str(np.random.choice(["kmeans", "k-means++", "random_from_data"])),
        }

        for i in range(args.num_runs_per_perturbation):
            run_random_force(
                logging_path=os.path.join(perturb_path, f"run_{i:04d}"),
                params=experiment_specification,
                **experiment_specification["script"]["args"],
            )


if __name__ == "__main__":
    main()
