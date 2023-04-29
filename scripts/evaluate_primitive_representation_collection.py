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
import copy

import wandb

from sim2sim.util.script_utils import rank_based_on_final_errors

QUATERNION_ERROR_WEIGHT = 2.0
TRANSLATION_ERROR_WEIGHT = 5.0
ANGULAR_VELOCITY_ERROR_WEIGHT = 1.0
TRANSLATIONAL_VELOCITY_ERROR_WEIGHT = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="The path to the primitive representation collection.",
    )
    parser.add_argument(
        "--experiment_description",
        required=True,
        type=str,
        help="The path to the experiment description. The description must use "
        + "'IdentityPrimitiveMeshProcessor' for 'inner_mesh_processor' and "
        + "'GTPhysicalPropertyEstimator' for both 'outer_physical_property_estimator' "
        + "and 'inner_physical_property_estimator'.",
    )
    parser.add_argument(
        "--logging_path",
        required=False,
        type=str,
        help="The path to log the results to. A temporary folder will be created and " + "deleted if not given.",
    )
    args = parser.parse_args()
    representation_collection_path = args.path
    experiment_description_path = args.experiment_description
    logging_path = args.logging_path

    base_experiment_description = yaml.safe_load(open(experiment_description_path, "r"))

    current_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    wandb.init(
        project="sim2sim_evaluate_representation_collection",
        name=f"{base_experiment_description['experiment_id']}_{current_time}",
        config=vars(args),
    )

    logging_path_is_tmp = False
    if not logging_path:
        logging_path = "logs/tmp_evaluate_primitive_representation_collection_folder"
        logging_path_is_tmp = True

    if not os.path.exists(logging_path):
        print(f"Creating folder {logging_path}")
        os.mkdir(logging_path)

    physical_properties_path = os.path.join(representation_collection_path, "physical_properties.yaml")
    pysical_properties = yaml.safe_load(open(physical_properties_path, "r"))

    experiment_specifications: List[dict] = []
    with os.scandir(representation_collection_path) as paths:
        for path in paths:
            if path.is_dir():
                experiment_description = copy.deepcopy(base_experiment_description)
                experiment_description["experiment_id"] = path.name

                # Primitive info
                experiment_description["inner_mesh_processor"]["args"]["primitive_info_path"] = os.path.join(
                    path.path, "primitive_info.pkl"
                )

                # Outer physical properties
                experiment_description["outer_physical_property_estimator"]["args"]["mass"] = pysical_properties["mass"]
                experiment_description["outer_physical_property_estimator"]["args"]["inertia"] = pysical_properties[
                    "inertia"
                ]

                # Inner physical properties
                experiment_description["inner_physical_property_estimator"]["args"]["mass"] = pysical_properties["mass"]
                experiment_description["inner_physical_property_estimator"]["args"]["inertia"] = pysical_properties[
                    "inertia"
                ]

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