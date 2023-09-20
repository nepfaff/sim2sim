"""Entrypoint for running an experiment."""

#!/bin/python3
import argparse

import yaml

from sim2sim.experiments import (
    run_floor_drop,
    run_iiwa_manip,
    run_planar_pushing,
    run_random_force,
    run_table_pid,
)

ENTRYPOINTS = {
    "table_pid": run_table_pid,
    "iiwa_manip": run_iiwa_manip,
    "floor_drop": run_floor_drop,
    "random_force": run_random_force,
    "planar_pushing": run_planar_pushing,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_description",
        required=True,
        type=str,
        help="The path to the yaml file that descripes the experiment to run.",
    )
    args = parser.parse_args()

    experiment_specification = yaml.safe_load(open(args.experiment_description, "r"))

    runner = ENTRYPOINTS[experiment_specification["script"]["name"]]
    runner(
        logging_path=f"logs/{experiment_specification['experiment_id']}",
        params=experiment_specification,
        **experiment_specification["script"]["args"],
    )


if __name__ == "__main__":
    main()
