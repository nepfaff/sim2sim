"""
Script for collecting sphere pushing data for metric learning.
NOTE: Using textured meshes in inverse graphics/ mesh processing might cause issues with multicore execution.
"""

import os
import yaml
import argparse
import glob
import re
from multiprocessing import Process

import numpy as np
from tqdm import tqdm
import open3d as o3d

from sim2sim.experiments import run_sphere_pushing
from sim2sim.images import generate_camera_locations_circle

PERTURBATION_DIR_BASENAME = "perturb_"
PERTURBATION_RANDOM_SEED = 1
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
        "--num_runs_per_perturbation",
        default=10,
        type=int,
        help="How many random force experiments to run per mesh perturbation.",
    )
    parser.add_argument(
        "--num_perturbations",
        default=1000,
        type=int,
        help="The number of different perturbations to run.",
    )
    parser.add_argument(
        "--viz_starting_locations",
        action="store_true",
        help="Whether to visualize the possible sphere starting locations.",
    )
    args = parser.parse_args()

    experiment_specification = yaml.safe_load(open(args.experiment_description, "r"))

    # Find start perturbation number to prevent overwritting already generated data
    perturb_num = -1
    if os.path.exists(args.logging_path):
        for path in glob.iglob(f"{args.logging_path}/{PERTURBATION_DIR_BASENAME}*"):
            num = int(re.findall(r"\d+", path)[-1])
            if num > perturb_num:
                perturb_num = num
    else:
        os.mkdir(args.logging_path)

    sphere_start_locations = []
    for z_distance, radius, num_locations in zip(
        [0.06, 0.12, 0.18], [0.3, 0.25, 0.2], [10, 10, 10]
    ):
        locations = generate_camera_locations_circle(
            center=[0.0, 0.0, z_distance],
            radius=radius,
            num_points=num_locations,
            xz=False,
        )
        sphere_start_locations.append(locations)
    sphere_start_locations = np.concatenate(sphere_start_locations, axis=0)

    if args.viz_starting_locations:
        viz_geoms = []
        for location in sphere_start_locations:
            viz_geoms.append(
                o3d.geometry.TriangleMesh.create_sphere(0.01).translate(location)
            )
        o3d.visualization.draw_geometries(viz_geoms)

    perturbation_rng = np.random.default_rng(PERTURBATION_RANDOM_SEED)
    for _ in tqdm(range(args.num_perturbations)):
        perturb_num += 1

        perturb_path = os.path.join(
            args.logging_path, f"{PERTURBATION_DIR_BASENAME}{perturb_num:06d}"
        )
        os.mkdir(perturb_path)

        # Random perturbation
        experiment_specification["mesh_processor"]["args"]["gmm_em_params"] = {
            "n_components": int(perturbation_rng.uniform(1, 150)),
            "tol": perturbation_rng.uniform(0.0, 0.01),
            "max_iter": int(perturbation_rng.normal(100, 15)),
            "n_init": 1 + int(perturbation_rng.choice(5)),
            "init_params": str(
                perturbation_rng.choice(["kmeans", "k-means++", "random_from_data"])
            ),
        }

        random_seed = START_RANDOM_SEED
        processes = []
        for i in range(args.num_runs_per_perturbation):
            rng = np.random.default_rng(random_seed)
            sphere_start_location = sphere_start_locations[
                rng.choice(len(sphere_start_locations))
            ]
            experiment_specification["script"]["args"][
                "sphere_starting_position"
            ] = sphere_start_location
            random_seed += 1

            kwargs = {
                "logging_path": os.path.join(perturb_path, f"run_{i:04d}"),
                "params": experiment_specification,
            }
            kwargs.update(experiment_specification["script"]["args"])
            p = Process(target=run_sphere_pushing, kwargs=kwargs)
            processes.append(p)
            p.start()

        for process in processes:
            process.join()


if __name__ == "__main__":
    main()
