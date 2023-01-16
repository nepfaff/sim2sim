"""Script for collecting multiple random force data runs for a representation."""

import os
import yaml
import argparse
from multiprocessing import Process

from sim2sim.experiments import run_random_force


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
    args = parser.parse_args()

    experiment_specification = yaml.safe_load(open(args.experiment_description, "r"))

    if not os.path.exists(args.logging_path):
        os.mkdir(args.logging_path)

    for i in range(args.num_runs):
        processes = []
        kwargs = {
            "logging_path": os.path.join(args.logging_path, f"run_{i:04d}"),
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
