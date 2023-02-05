"""Script for visualizing a collision representation and its contact forces at a given simulation time point."""

import argparse
import os
import yaml
import time

import numpy as np
from pydrake.all import (
    StartMeshcat,
    MeshcatVisualizerParams,
    Role,
    Meshcat,
    MeshcatVisualizer,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Simulator,
    LoadModelDirectives,
    ProcessModelDirectives,
    Rgba,
    Cylinder,
    MeshcatCone,
    RigidTransform,
    RotationMatrix,
)

from sim2sim.util import get_parser


def loda_data(log_dir: str, outer: bool):
    prefix = "outer" if outer else "inner"

    def process_array(arr):
        max_count = 0
        for el in arr:
            max_count = max(max_count, len(el))

        return np.array(
            [
                np.concatenate([el, np.zeros((max_count - len(el), 3))], axis=0)
                if len(el) > 0
                else np.zeros((max_count, 3))
                for el in arr
            ]
        )

    # Hydroelastic
    contact_result_force_centroids_raw = np.load(
        os.path.join(log_dir, f"{prefix}_hydroelastic_contact_result_centroids.npy"),
        allow_pickle=True,
    )
    hydroelastic_centroids = process_array(contact_result_force_centroids_raw)
    contact_result_forces_raw = np.load(
        os.path.join(log_dir, f"{prefix}_hydroelastic_contact_result_forces.npy"),
        allow_pickle=True,
    )
    hydroelastic_contact_forces = process_array(contact_result_forces_raw)

    # Point contact
    point_contact_result_contact_points_raw = np.load(
        os.path.join(log_dir, f"{prefix}_point_contact_result_contact_points.npy"), allow_pickle=True
    )
    point_contact_points = process_array(point_contact_result_contact_points_raw)
    point_contact_result_forces_raw = np.load(
        os.path.join(log_dir, f"{prefix}_point_contact_result_forces.npy"), allow_pickle=True
    )
    point_contact_forces = process_array(point_contact_result_forces_raw)

    # Manipuland poses
    manipuland_poses = np.loadtxt(os.path.join(log_dir, f"{prefix}_manipuland_poses.txt"))[:, 7]

    return (
        hydroelastic_centroids,
        hydroelastic_contact_forces,
        point_contact_points,
        point_contact_forces,
        manipuland_poses,
    )


def add_arrow(
    meshcat: Meshcat,
    path: str,
    force: np.ndarray,
    contact_point: np.ndarray,
    radius: float = 0.001,
    rgba=Rgba(1.0, 0.0, 0.0, 1.0),
) -> None:

    force_norm = np.linalg.norm(force)
    newtons_per_meter = 1e2  # TODO: Make an argument

    # Create arrow
    height = force_norm / newtons_per_meter
    # Cylinder gets scaled to twice the contact force length because we draw both (equal and opposite) forces
    cylinder = Cylinder(radius, 2 * height)
    meshcat.SetObject(
        path=path + "/cylinder",
        shape=cylinder,
        rgba=rgba,
    )
    arrowhead_height = arrowhead_width = radius * 2.0
    arrowhead = MeshcatCone(arrowhead_height, arrowhead_width, arrowhead_width)
    meshcat.SetObject(
        path=path + "/head",
        shape=arrowhead,
        rgba=rgba,
    )
    meshcat.SetObject(
        path=path + "/tail",
        shape=arrowhead,
        rgba=rgba,
    )

    # Transform arrow
    meshcat.SetTransform(
        path, RigidTransform(RotationMatrix.MakeFromOneVector(force, 2), contact_point)
    )  # Arrow starts along z-axis (axis 2)
    meshcat.SetTransform(path + "/head", RigidTransform([0.0, 0.0, -height - arrowhead_height]))
    meshcat.SetTransform(
        path + "/tail", RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.0, 0.0, height + arrowhead_height])
    )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to the experiment data folder.",
    )
    arg_parser.add_argument(
        "--time",  # TODO: Allow stepping through time with meshcat slider
        required=True,
        type=float,
        help="The simulation time to visualize the contact forces for.",
    )
    arg_parser.add_argument(
        "--hydroelastic", action="store_true", help="Whether to plot hydroelastic or point contact forces."
    )
    arg_parser.add_argument(
        "--manipuland",
        default="outer",
        type=str,
        help="The manipuland to visualize. Options are 'outer', 'inner', 'both', and 'none'.",
    )
    args = arg_parser.parse_args()

    assert args.manipuland in ["outer", "inner", "both", "none"]

    log_dir = os.path.join(args.data, "time_logs")
    times = np.loadtxt(os.path.join(log_dir, "outer_manipuland_pose_times.txt"))
    (
        outer_hydroelastic_centroids,
        outer_hydroelastic_contact_forces,
        outer_point_contact_points,
        outer_point_contact_forces,
        outer_manipuland_poses,
    ) = loda_data(log_dir, outer=True)
    (
        inner_hydroelastic_centroids,
        inner_hydroelastic_contact_forces,
        inner_point_contact_points,
        inner_point_contact_forces,
        inner_manipuland_poses,
    ) = loda_data(log_dir, outer=False)

    # TODO: Visualize both outer and inner forces on object in one simulation (argument to choose which object to visualize)
    # TODO: Different colors for outer and inner forces

    time_idx = np.abs(times - args.time).argmin()
    time_diff = abs(times[time_idx] - args.time)
    if time_diff > 0.01:
        print(f"Demanded and available time differ by {time_diff}s. Showing forces for time {times[time_idx]}s.")

    outer_hydroelastic_centroid = outer_hydroelastic_centroids[time_idx]
    outer_hydroelastic_contact_force = outer_hydroelastic_contact_forces[time_idx]
    print(f"outer_point_contact_point:\n{outer_point_contact_points[time_idx]}")
    print(f"outer_point_contact_force:\n{outer_point_contact_forces[time_idx]}")
    outer_point_contact_point = outer_point_contact_points[time_idx]
    outer_point_contact_force = outer_point_contact_forces[time_idx]
    outer_manipuland_pose = outer_manipuland_poses[time_idx]
    inner_hydroelastic_centroid = inner_hydroelastic_centroids[time_idx]
    inner_hydroelastic_contact_force = inner_hydroelastic_contact_forces[time_idx]
    print(f"inner_point_contact_point:\n{inner_point_contact_points[time_idx]}")
    print(f"inner_point_contact_force:\n{inner_point_contact_forces[time_idx]}")
    inner_point_contact_point = inner_point_contact_points[time_idx]
    inner_point_contact_force = inner_point_contact_forces[time_idx]
    inner_manipuland_pose = inner_manipuland_poses[time_idx]

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = get_parser(plant)

    # TODO: transform manipulands to pose (easier to transform with meshcat or
    # plant.SetDefaultFreeBodyPose(plant.GetBodyByName(manipuland_base_link_name), manipuland_pose)?)
    if args.manipuland in ["outer", "both"]:
        experiment_description_path = os.path.join(args.data, "experiment_description.yaml")
        experiment_description = yaml.safe_load(open(experiment_description_path, "r"))
        outer_manipuland_directive_path = os.path.join(
            args.data, experiment_description["script"]["args"]["manipuland_directive"]
        )
        outer_manipuland_directive = LoadModelDirectives(outer_manipuland_directive_path)
        ProcessModelDirectives(outer_manipuland_directive, parser)

    if args.manipuland in ["inner", "both"]:
        inner_manipuland_sdf_path = os.path.join(args.data, "meshes", "processed_mesh.sdf")
        parser.AddModelFromFile(inner_manipuland_sdf_path, "inner_manipuland")

    meshcat = StartMeshcat()
    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.role = Role.kProximity
    _ = MeshcatVisualizer.AddToBuilder(builder, scene_graph.get_query_output_port(), meshcat, meshcat_params)

    # TODO: Add hydroelastic option
    for i, (force, point) in enumerate(zip(outer_point_contact_force, outer_point_contact_point)):
        if np.linalg.norm(force) > 0.0:
            add_arrow(
                meshcat,
                path=f"contact_forces/outer_sim/force_{i}",
                force=force,
                contact_point=point,
                rgba=Rgba(0.0, 1.0, 0.0, 1.0),
            )

    for i, (force, point) in enumerate(zip(inner_point_contact_force, inner_point_contact_point)):
        if np.linalg.norm(force) > 0.0:
            add_arrow(
                meshcat,
                path=f"contact_forces/inner_sim/force_{i}",
                force=force,
                contact_point=point,
                rgba=Rgba(1.0, 0.0, 0.0, 1.0),
            )

    # inner_meshcat.Delete("contacts")

    plant.Finalize()
    diagram = builder.Build()

    # Need to simulate for visualization to work
    simulator = Simulator(diagram)
    simulator.AdvanceTo(0.0)

    # Sleep to give user enough time to click on meshcat link
    time.sleep(5.0)


if __name__ == "__main__":
    main()
