"""
Script for collecting data to test whether simulation time scales with collision geometry complexity or only with
contact points.
NOTE: Drake meshcat dies if more than 100 instances are spawned. Either commment out the meshcat code or run less than
this number of total runs.
"""

import os
import yaml
import argparse

from tqdm import tqdm

from sim2sim.experiments import run_random_force

PERTURBATION_DIR_BASENAME = "perturb_"
RANDOM_SEED = 101


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
        help="How many random force experiments to run per perturbation.",
    )
    parser.add_argument(
        "--num_perturbations", default=40, type=int, help="The number of different perturbations (GMM nums) to run."
    )
    args = parser.parse_args()

    experiment_specification = yaml.safe_load(open(args.experiment_description, "r"))

    if not os.path.exists(args.logging_path):
        os.mkdir(args.logging_path)

    gmm_num = 1
    for _ in tqdm(range(args.num_perturbations)):
        perturb_path = os.path.join(args.logging_path, f"{PERTURBATION_DIR_BASENAME}{gmm_num:06d}")
        os.mkdir(perturb_path)

        # Perturb GMM num
        experiment_specification["mesh_processor"]["args"]["gmm_em_params"]["n_components"] = gmm_num

        for i in range(args.num_runs_per_perturbation):
            experiment_specification["simulator"]["args"]["random_seed"] = RANDOM_SEED
            run_random_force(
                logging_path=os.path.join(perturb_path, f"run_{i:04d}"),
                params=experiment_specification,
                **experiment_specification["script"]["args"],
            )

        gmm_num += 1


if __name__ == "__main__":
    main()
