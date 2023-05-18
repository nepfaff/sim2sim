"""Entrypoint for running an experiment."""

#!/bin/python3
from trimesh.viewer import SceneViewer  # https://github.com/mikedh/trimesh/issues/1225

import yaml
import argparse

from sim2sim.experiments import (
    run_table_pid,
    run_iiwa_manip_tomato_soup_can,
    run_iiwa_manip_pepper,
    run_floor_drop,
    run_random_force,
    run_planar_pushing,
)

ENTRYPOINTS = {
    "table_pid": run_table_pid,
    "iiwa_manip_tomato_soup_can": run_iiwa_manip_tomato_soup_can,
    "iiwa_manip_pepper": run_iiwa_manip_pepper,
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
