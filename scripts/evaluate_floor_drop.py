"""Script for producing floor drop experiment data for multiple meshes."""

import os
import yaml
import argparse
import glob
from typing import List, Dict

import numpy as np
from prettytable import PrettyTable

from sim2sim.experiments import run_floor_drop

QUATERNION_ERROR_WEIGHT = 2.0
TRANSLATION_ERROR_WEIGHT = 5.0
ANGULAR_VELOCITY_ERROR_WEIGHT = 1.0
TRANSLATIONAL_VELOCITY_ERROR_WEIGHT = 1.0


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
        help="The path that contains the meshes to collect data for. Any primitive info files corresponding to these "
        + "meshes should be in the same directory and have the same name (but .pkl instead of .obj extension). "
        + "This must be an absolute path.",
    )
    parser.add_argument(
        "--logging_path",
        required=True,
        type=str,
        help="The path to log the results to.",
    )
    args = parser.parse_args()

    # TODO: Add options to parallelize this
    eval_data: List[Dict[str, float]] = []
    for mesh_path in glob.iglob(f"{args.meshes}/*.obj"):
        mesh_name = os.path.split(mesh_path)[-1][:-4]
        print(f"\nEvaluating {mesh_name}:")

        experiment_specification = yaml.safe_load(open(args.experiment_description, "r"))
        experiment_specification["inverse_graphics"]["args"]["mesh_path"] = mesh_path

        mesh_processor_name = experiment_specification["mesh_processor"]["class"]
        if mesh_processor_name == "CoACDMeshProcessor":
            experiment_specification["mesh_processor"]["args"]["mesh_name"] = mesh_path
        elif mesh_processor_name == "ConvexDecompMeshProcessor":
            experiment_specification["mesh_processor"]["args"]["mesh_name"] = mesh_path
        elif mesh_processor_name == "FuzzyMetaballMeshProcessor":
            experiment_specification["mesh_processor"]["args"]["mesh_path"] = mesh_path
        elif mesh_processor_name == "IdentityPrimitiveMeshProcessor":
            primitive_info_file_path = mesh_path[:-4] + ".pkl"
            assert os.path.exists(primitive_info_file_path), f"Mesh {mesh_name} has no associated primitive info file."
            experiment_specification["mesh_processor"]["args"]["primitive_info_path"] = primitive_info_file_path

        logging_path = os.path.join(args.logging_path, mesh_name)
        run_floor_drop(
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
                "name": mesh_name,
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
            f"Mesh: {mesh_name}, Translation err: {translation_error}, Quaternion err: {quaternion_error}, "
            + f"Translational velocity err: {translational_velocity_error}, Angular velocity err: {angular_velocity_error}"
        )

    table = create_evaluation_results_table(eval_data)
    print("\n\nEvaluation results:\n")
    print(table)


if __name__ == "__main__":
    main()
