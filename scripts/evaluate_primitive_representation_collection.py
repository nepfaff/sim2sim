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
import json
import random

import wandb

from sim2sim.util.script_utils import rank_based_on_metrics


def make_outer_deterministic(
    experiment_specifications: List[dict],
    experiment_description: dict,
    logging_path: str,
    skip_outer_visualization: bool,
) -> dict:
    """
    Ensure that the outer sim is deterministic by re-using the SDFormat file from the
    first experiment.
    """
    first_experiment_name = experiment_specifications[0]["experiment_id"]
    outer_sdf_path = os.path.join(
        logging_path,
        first_experiment_name,
        "meshes/outer_processed_mesh.sdf",
    )
    experiment_description["outer_mesh_processor"]["class"] = "IdentitySDFMeshProcessor"
    experiment_description["outer_mesh_processor"]["args"] = {
        "sdf_path": outer_sdf_path
    }

    # No need to visualize outer sim as will be a copy of the first visualization
    experiment_description["simulator"]["args"]["skip_outer_visualization"] = True

    return experiment_description


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
        help="The path to log the results to. A temporary folder will be created and "
        + "deleted if not given.",
    )
    parser.add_argument(
        "--num_trajectory_iou_samples",
        default=500,
        type=int,
        help="The number of samples to use per trajectory state for trajectory IoU "
        + "metric computation. More samples results in more accurate metrics but is "
        + "slower to compute.",
    )
    parser.add_argument(
        "--eval_contact_model",
        action="store_true",
        help="Whether to evaluate both 'hydroelastic_with_fallback' and 'point' contact "
        + "models for the primitive-based representations. Otherwise, the one from the "
        + "experiment description will be used.",
    )
    parser.add_argument(
        "--additional_experiment_descriptions",
        required=False,
        default=None,
        type=json.loads,
        help="The paths to additional experiment descriptions to include in the "
        + "evaluation. These experiment descriptions will be added without modifications "
        + "(apart from making the outer sim deterministic). Hence, they should be "
        + "similar enough to '--experiment_description' to allow for fair comparison.",
    )
    parser.add_argument(
        "--keep_outer_vis",
        action="store_true",
        help="Whether to create a visualizer for each outer simulation. If False, only "
        + "the first outer simulation is visualized/ recorded.",
    )
    parser.add_argument(
        "--wandb_name",
        required=False,
        default=None,
        type=str,
        help="An optional custom wandb name",
    )

    start_time = time.time()

    args = parser.parse_args()
    representation_collection_path = args.path
    experiment_description_path = args.experiment_description
    logging_path = args.logging_path
    additional_experiment_descriptions = args.additional_experiment_descriptions
    skip_outer_visualization = not args.keep_outer_vis
    wandb_name = args.wandb_name

    base_experiment_description = yaml.safe_load(open(experiment_description_path, "r"))

    current_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    wandb.init(
        project="sim2sim_evaluate_representation_collection",
        name=f"{base_experiment_description['experiment_id']}_{current_time}"
        if wandb_name is None
        else wandb_name,
        config={
            "args": vars(args),
            "base_experiment_description": base_experiment_description,
        },
    )

    logging_path_is_tmp = False
    if not logging_path:
        logging_path = "logs/tmp_evaluate_primitive_representation_collection_folder"
        logging_path_is_tmp = True

    if not os.path.exists(logging_path):
        print(f"Creating folder {logging_path}")
        os.mkdir(logging_path)

    physical_properties_path = os.path.join(
        representation_collection_path, "physical_properties.yaml"
    )
    pysical_properties = yaml.safe_load(open(physical_properties_path, "r"))

    # Add additional experiment descriptions
    experiment_specifications: List[dict] = []
    if additional_experiment_descriptions is not None:
        for additional_description_path in additional_experiment_descriptions:
            additional_description = yaml.safe_load(
                open(additional_description_path, "r")
            )
            if len(experiment_specifications) > 0:
                additional_description = make_outer_deterministic(
                    experiment_specifications,
                    additional_description,
                    logging_path,
                    skip_outer_visualization,
                )
            experiment_specifications.append(additional_description)

    with os.scandir(representation_collection_path) as paths:
        for path in paths:
            if path.is_dir():
                experiment_description = copy.deepcopy(base_experiment_description)
                experiment_description["experiment_id"] = path.name

                if len(experiment_specifications) > 0:
                    experiment_description = make_outer_deterministic(
                        experiment_specifications,
                        experiment_description,
                        logging_path,
                        skip_outer_visualization,
                    )

                # Primitive info
                experiment_description["inner_mesh_processor"]["args"][
                    "primitive_info_path"
                ] = os.path.join(path.path, "primitive_info.pkl")

                # Outer physical properties
                experiment_description["outer_physical_property_estimator"]["args"][
                    "mass"
                ] = pysical_properties["mass"]
                experiment_description["outer_physical_property_estimator"]["args"][
                    "inertia"
                ] = pysical_properties["inertia"]
                experiment_description["outer_physical_property_estimator"]["args"][
                    "center_of_mass"
                ] = pysical_properties["com"]

                # Inner physical properties
                experiment_description["inner_physical_property_estimator"]["args"][
                    "mass"
                ] = pysical_properties["mass"]
                experiment_description["inner_physical_property_estimator"]["args"][
                    "inertia"
                ] = pysical_properties["inertia"]
                experiment_description["inner_physical_property_estimator"]["args"][
                    "center_of_mass"
                ] = pysical_properties["com"]

                if args.eval_contact_model:
                    hydroelastic_experiment_description = copy.deepcopy(
                        experiment_description
                    )
                    hydroelastic_experiment_description[
                        "experiment_id"
                    ] += "_hydroelastic"
                    hydroelastic_experiment_description["inner_env"][
                        "contact_model"
                    ] = "hydroelastic_with_fallback"
                    experiment_specifications.append(
                        hydroelastic_experiment_description
                    )

                    point_experiment_description = copy.deepcopy(experiment_description)
                    point_experiment_description["experiment_id"] += "_point"
                    point_experiment_description["inner_env"]["contact_model"] = "point"
                    experiment_specifications.append(point_experiment_description)
                else:
                    experiment_specifications.append(experiment_description)

    # Shuffle experiment order for fairer runtime estimates
    # Need to keep the first first due to deterministic outer
    first_specification = experiment_specifications[0]
    other_specifications = experiment_specifications[1:]
    random.shuffle(other_specifications)
    experiment_specifications = [first_specification, *other_specifications]

    print(f"Running {len(experiment_specifications)} experiments.")

    rank_based_on_metrics(
        experiment_specifications,
        logging_path,
        num_trajectory_iou_samples=args.num_trajectory_iou_samples,
        log_wandb=True,
    )

    if logging_path_is_tmp:
        print(f"Cleaning up temporary logging folder {logging_path}")
        shutil.rmtree(logging_path)

    print(f"Evaluation took {(time.time()-start_time)/60.0} minutes.")


if __name__ == "__main__":
    main()
