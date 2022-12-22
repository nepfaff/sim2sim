"""Script for producing floor drop experiment data for multiple meshes."""

import os
import yaml
import argparse
import glob

from sim2sim.experiments import run_floor_drop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_description",
        required=True,
        type=str,
        help="The path to the yaml file that descripes the experiment to run.",
    )
    parser.add_argument(
        "--meshes",
        required=True,
        type=str,
        help="The path that contains the meshes to collect data for.",
    )
    parser.add_argument(
        "--logging_path",
        required=True,
        type=str,
        help="The path to log the results to.",
    )
    args = parser.parse_args()

    for mesh_path in glob.iglob(f"{args.meshes}/*.obj"):
        experiment_specification = yaml.safe_load(open(args.experiment_description, "r"))
        experiment_specification["inverse_graphics"]["args"]["mesh_path"] = mesh_path

        mesh_name = os.path.split(mesh_path)[-1][:-4]
        print(f"\nEvaluating {mesh_name}:")

        run_floor_drop(
            logging_path=os.path.join(args.logging_path, mesh_name),
            params=experiment_specification,
            **experiment_specification["script"]["args"],
        )


if __name__ == "__main__":
    main()
