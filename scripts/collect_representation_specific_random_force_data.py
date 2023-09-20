"""
Script for collecting multiple random force data runs for a representation.
NOTE: Using textured meshes in inverse graphics/ mesh processing might cause issues with multicore execution.
"""

import argparse
import os

from multiprocessing import Process

import yaml

from sim2sim.experiments import run_random_force

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
        "--num_runs",
        default=10,
        type=int,
        help="How many random force experiments to run.",
    )
    parser.add_argument(
        "--use_random_seed",
        action="store_true",
        help="Whether to use a random seed for repeatability.",
    )
    parser.add_argument(
        "--single_core",
        action="store_true",
        help="Whether to use a single core.",
    )
    args = parser.parse_args()

    experiment_specification = yaml.safe_load(open(args.experiment_description, "r"))

    if not os.path.exists(args.logging_path):
        os.mkdir(args.logging_path)

    random_seed = START_RANDOM_SEED
    processes = []
    for i in range(args.num_runs):
        if args.use_random_seed:
            experiment_specification["simulator"]["args"]["random_seed"] = random_seed
            random_seed += 1

        if args.single_core:
            run_random_force(
                logging_path=os.path.join(args.logging_path, f"run_{i:04d}"),
                params=experiment_specification,
                **experiment_specification["script"]["args"],
            )
        else:
            kwargs = {
                "logging_path": os.path.join(args.logging_path, f"run_{i:04d}"),
                "params": experiment_specification,
            }
            kwargs.update(experiment_specification["script"]["args"])
            p = Process(target=run_random_force, kwargs=kwargs)
            processes.append(p)
            p.start()

    if not args.single_core:
        for process in processes:
            process.join()


if __name__ == "__main__":
    main()
