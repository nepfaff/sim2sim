"""
Script for collecting random force data for metric learning.
NOTE: Using textured meshes in inverse graphics/ mesh processing might cause issues with multicore execution.
"""

import os
import yaml
import argparse
import glob
import re
from multiprocessing import Process

import numpy as np
from tqdm import tqdm

from sim2sim.experiments import run_random_force

PERTURBATION_DIR_BASENAME = "perturb_"
PERTURBATION_RANDOM_SEED = 1
START_RANDOM_SEED = 100


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
        "--num_perturbations",
        default=1000,
        type=int,
        help="The number of different perturbations to run.",
    )
    parser.add_argument(
        "--use_random_seed",
        action="store_true",
        help="Whether to use a random seed for repeatability.",
    )
    args = parser.parse_args()

    experiment_specification = yaml.safe_load(open(args.experiment_description, "r"))

    # Find start perturbation number to prevent overwritting already generated data
    perturb_num = -1
    if os.path.exists(args.logging_path):
        for path in glob.iglob(f"{args.logging_path}/{PERTURBATION_DIR_BASENAME}*"):
            num = int(re.findall(r"\d+", path)[-1])
            if num > perturb_num:
                perturb_num = num
    else:
        os.mkdir(args.logging_path)

    perturbation_rng = np.random.default_rng(PERTURBATION_RANDOM_SEED)
    for _ in tqdm(range(args.num_perturbations)):
        perturb_num += 1

        perturb_path = os.path.join(
            args.logging_path, f"{PERTURBATION_DIR_BASENAME}{perturb_num:06d}"
        )
        os.mkdir(perturb_path)

        # Random perturbation
        experiment_specification["mesh_processor"]["args"]["gmm_em_params"] = {
            "n_components": int(perturbation_rng.uniform(1, 150)),
            "tol": perturbation_rng.uniform(0.0, 0.01),
            "max_iter": int(perturbation_rng.normal(100, 15)),
            "n_init": 1 + int(perturbation_rng.choice(5)),
            "init_params": str(
                perturbation_rng.choice(["kmeans", "k-means++", "random_from_data"])
            ),
        }

        random_seed = START_RANDOM_SEED
        processes = []
        for i in range(args.num_runs_per_perturbation):
            if args.use_random_seed:
                experiment_specification["simulator"]["args"][
                    "random_seed"
                ] = random_seed
                random_seed += 1
            kwargs = {
                "logging_path": os.path.join(perturb_path, f"run_{i:04d}"),
                "params": experiment_specification,
            }
            kwargs.update(experiment_specification["script"]["args"])
            p = Process(target=run_random_force, kwargs=kwargs)
            processes.append(p)
            p.start()

        for process in processes:
            process.join()


if __name__ == "__main__":
    main()
