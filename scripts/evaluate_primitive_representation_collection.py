"""
Script for ranking different real2sim approaches based on final errors in translation,
rotation, and velocity.

NOTE: Each approach is described completely with an experiment description file. For
fair comparison, the experiment
should be the same between files with some of the real2sim pipeline components being
different.
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
import re

import wandb

from sim2sim.util.script_utils import rank_based_on_metrics


def make_outer_deterministic(
    experiment_specifications: List[dict],
    experiment_description: dict,
    logging_path: str,
    num_manipulands: int,
    skip_outer_visualization: bool,
) -> dict:
    """
    Ensure that the outer sim is deterministic by re-using the SDFormat file from the
    first experiment.
    """
    first_experiment_name = experiment_specifications[0]["experiment_id"]
    experiment_description["outer_mesh_processor"]["class"] = "IdentitySDFMeshProcessor"

    outer_sdf_paths = [
        os.path.join(
            logging_path,
            first_experiment_name,
            "meshes",
            f"outer_processed_mesh_{i}.sdf",
        )
        for i in range(num_manipulands)
    ]
    experiment_description["outer_mesh_processor"]["args"] = {
        "sdf_paths": outer_sdf_paths
    }

    # No need to visualize outer sim as will be a copy of the first visualization
    if experiment_description["simulator"]["args"] is None:
        experiment_description["simulator"]["args"] = {
            "skip_outer_visualization": skip_outer_visualization
        }
    else:
        experiment_description["simulator"]["args"][
            "skip_outer_visualization"
        ] = skip_outer_visualization

    return experiment_description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        required=True,
        type=json.loads,
        help="The paths to the primitive representation collections. The first "
        + "manipuland will be read from the first item in the list and so on.",
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
        "--additional_collision_geometries_paths",
        required=False,
        default=None,
        type=json.loads,
        help="The paths to folders containing additional collision geometries to "
        + "include in the evaluation. These collision geometries will be added "
        + "similarly to the primitive collision geometries, by modidfying the provided "
        + "experiment description. The first manipuland will be read from the first "
        + "item in the list and so on.",
    )
    parser.add_argument(
        "--additional_collision_geometries_mesh_pieces_paths",
        required=False,
        default=None,
        type=json.loads,
        help="The paths to folders containing additional collision geometries to "
        + "include in the evaluation. The collision geometries are represented by a "
        + "folder, containing multiple mesh pieces. The individual pieces make up one "
        + "collision geometry. These collision geometries will be added "
        + "similarly to the primitive collision geometries, by modidfying the provided "
        + "experiment description. The first manipuland will be read from the first "
        + "item in the list and so on.",
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
    representation_collection_paths = args.paths
    experiment_description_path = args.experiment_description
    logging_path = args.logging_path
    additional_experiment_descriptions = args.additional_experiment_descriptions
    additional_collision_geometries_paths = args.additional_collision_geometries_paths
    additional_collision_geometries_mesh_pieces_paths = (
        args.additional_collision_geometries_mesh_pieces_paths
    )
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

    physical_properties_paths = [
        os.path.join(path, "physical_properties.yaml")
        for path in representation_collection_paths
    ]
    pysical_properties = [
        yaml.safe_load(open(path, "r")) for path in physical_properties_paths
    ]

    # Add additional experiment descriptions
    experiment_specifications: List[dict] = []
    if additional_experiment_descriptions is not None:
        for additional_description_path in additional_experiment_descriptions:
            additional_description = yaml.safe_load(
                open(additional_description_path, "r")
            )
            if len(experiment_specifications) > 0:
                additional_description = make_outer_deterministic(
                    experiment_specifications=experiment_specifications,
                    experiment_description=additional_description,
                    logging_path=logging_path,
                    num_manipulands=len(representation_collection_paths),
                    skip_outer_visualization=skip_outer_visualization,
                )
            experiment_specifications.append(additional_description)

    def adjust_experiment_description(experiment_description: dict) -> List[dict]:
        """Make representation collection specific edits to the experiment description."""

        masses = []
        inertias = []
        centers_of_mass = []
        for properties in pysical_properties:
            masses.append(properties["mass"])
            inertias.append(properties["inertia"])
            centers_of_mass.append(properties["com"])

        # Outer physical properties
        experiment_description["outer_physical_property_estimator"]["args"][
            "masses"
        ] = masses
        experiment_description["outer_physical_property_estimator"]["args"][
            "inertias"
        ] = inertias
        experiment_description["outer_physical_property_estimator"]["args"][
            "centers_of_mass"
        ] = centers_of_mass

        # Inner physical properties
        experiment_description["inner_physical_property_estimator"]["args"][
            "masses"
        ] = masses
        experiment_description["inner_physical_property_estimator"]["args"][
            "inertias"
        ] = inertias
        experiment_description["inner_physical_property_estimator"]["args"][
            "centers_of_mass"
        ] = centers_of_mass

        adjusted_descriptions = []
        if args.eval_contact_model:
            hydroelastic_experiment_description = copy.deepcopy(experiment_description)
            hydroelastic_experiment_description["experiment_id"] += "_hydroelastic"
            hydroelastic_experiment_description["inner_env"][
                "contact_model"
            ] = "hydroelastic_with_fallback"
            adjusted_descriptions.append(hydroelastic_experiment_description)

            point_experiment_description = copy.deepcopy(experiment_description)
            point_experiment_description["experiment_id"] += "_point"
            point_experiment_description["inner_env"]["contact_model"] = "point"
            adjusted_descriptions.append(point_experiment_description)
        else:
            adjusted_descriptions.append(experiment_description)

        return adjusted_descriptions

    # Add additional collision geometries
    if additional_collision_geometries_paths is not None:
        with os.scandir(additional_collision_geometries_paths[0]) as paths:
            for path in paths:
                if path.is_file():
                    experiment_description = copy.deepcopy(base_experiment_description)
                    experiment_description["experiment_id"] = os.path.splitext(
                        path.name
                    )[0]

                    if len(experiment_specifications) > 0:
                        experiment_description = make_outer_deterministic(
                            experiment_specifications=experiment_specifications,
                            experiment_description=experiment_description,
                            logging_path=logging_path,
                            num_manipulands=len(representation_collection_paths),
                            skip_outer_visualization=skip_outer_visualization,
                        )

                    # Add collision geometries
                    mesh_paths = [path.path]
                    for (
                        additional_collision_geometries_path
                    ) in additional_collision_geometries_paths[1:]:
                        mesh_path = os.path.join(
                            additional_collision_geometries_path, path.name
                        )
                        assert os.path.exists(mesh_path), (
                            f"{path.name} does not exist in "
                            + f"{additional_collision_geometries_path}! Note that all "
                            + "additional collision geometries paths must contain "
                            + "directories with the same names that correspond to "
                            + "equivalent representations for different manipulands."
                        )
                        mesh_paths.append(mesh_path)

                    experiment_description["inner_inverse_graphics"]["args"][
                        "mesh_paths"
                    ] = mesh_paths
                    experiment_description["inner_mesh_processor"][
                        "class"
                    ] = "IdentityMeshProcessor"
                    experiment_description["inner_mesh_processor"]["args"] = {}

                    # Make edits based on representation collection
                    adjusted_experiment_specifications = adjust_experiment_description(
                        experiment_description
                    )
                    experiment_specifications.extend(adjusted_experiment_specifications)

    # Add additional mesh pieces based collision geometries
    if additional_collision_geometries_mesh_pieces_paths is not None:
        with os.scandir(additional_collision_geometries_mesh_pieces_paths[0]) as paths:
            for path in paths:
                if path.is_dir():
                    experiment_description = copy.deepcopy(base_experiment_description)
                    experiment_description["experiment_id"] = os.path.splitext(
                        path.name
                    )[0]

                    if len(experiment_specifications) > 0:
                        experiment_description = make_outer_deterministic(
                            experiment_specifications=experiment_specifications,
                            experiment_description=experiment_description,
                            logging_path=logging_path,
                            num_manipulands=len(representation_collection_paths),
                            skip_outer_visualization=skip_outer_visualization,
                        )

                    # Add collision geometries
                    mesh_pieces_paths = [path.path]
                    for (
                        additional_collision_geometries_mesh_pieces_path
                    ) in additional_collision_geometries_mesh_pieces_paths[1:]:
                        mesh_pieces_path = os.path.join(
                            additional_collision_geometries_mesh_pieces_path, path.name
                        )
                        assert os.path.exists(mesh_pieces_path), (
                            f"{path.name} does not exist in "
                            + f"{additional_collision_geometries_mesh_pieces_path}! "
                            + "Note that all additional collision geometries paths must "
                            + "contain directories with the same names that correspond "
                            + "to equivalent representations for different manipulands."
                        )
                        mesh_pieces_paths.append(mesh_pieces_path)

                    experiment_description["inner_mesh_processor"][
                        "class"
                    ] = "IdentityMeshPiecesMeshProcessor"
                    experiment_description["inner_mesh_processor"]["args"] = {}
                    experiment_description["inner_mesh_processor"]["args"][
                        "mesh_pieces_paths"
                    ] = mesh_pieces_paths

                    # Make edits based on representation collection
                    adjusted_experiment_specifications = adjust_experiment_description(
                        experiment_description
                    )
                    experiment_specifications.extend(adjusted_experiment_specifications)

    with os.scandir(representation_collection_paths[0]) as paths:
        for path in paths:
            if path.is_dir():
                experiment_description = copy.deepcopy(base_experiment_description)
                experiment_description["experiment_id"] = path.name

                # Write final SDF sample loss into description to enable inclusion into
                # metrics
                meta_data_path = os.path.join(path.path, "metadata.yaml")
                meta_data = yaml.safe_load(open(meta_data_path, "r"))
                experiment_description["sdf_loss"] = meta_data["final_sdf_loss"]

                if len(experiment_specifications) > 0:
                    experiment_description = make_outer_deterministic(
                        experiment_specifications=experiment_specifications,
                        experiment_description=experiment_description,
                        logging_path=logging_path,
                        num_manipulands=len(representation_collection_paths),
                        skip_outer_visualization=skip_outer_visualization,
                    )

                # Add primitive infos
                primitive_info_paths = [os.path.join(path.path, "primitive_info.pkl")]
                for representation_collection_path in representation_collection_paths[
                    1:
                ]:
                    primitive_info_path = os.path.join(
                        representation_collection_path, path.name, "primitive_info.pkl"
                    )
                    assert os.path.exists(primitive_info_path), (
                        f"{path.name} does not exist in {representation_collection_path}!"
                        + " Note that all representations collection paths must contain "
                        + "directories with the same names that correspond to "
                        + "equivalent representations for different manipulands."
                    )
                    primitive_info_paths.append(primitive_info_path)
                experiment_description["inner_mesh_processor"]["args"][
                    "primitive_info_paths"
                ] = primitive_info_paths

                # Make edits based on representation collection
                adjusted_experiment_specifications = adjust_experiment_description(
                    experiment_description
                )
                experiment_specifications.extend(adjusted_experiment_specifications)

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
        additional_metric_keys=["sdf_loss"],
    )

    if logging_path_is_tmp:
        print(f"Cleaning up temporary logging folder {logging_path}")
        shutil.rmtree(logging_path)

    print(f"Evaluation took {(time.time()-start_time)/60.0} minutes.")


if __name__ == "__main__":
    main()
