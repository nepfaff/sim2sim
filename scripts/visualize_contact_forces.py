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
    Quaternion,
)

from sim2sim.util import get_parser


def load_data(log_dir: str, outer: bool):
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
    manipuland_poses = np.loadtxt(os.path.join(log_dir, f"{prefix}_manipuland_poses.txt"))[:, :7]

    return (
        hydroelastic_centroids,
        hydroelastic_contact_forces,
        point_contact_points,
        point_contact_forces,
        manipuland_poses,
    )


def add_point_contact_arrow(
    meshcat: Meshcat,
    path: str,
    force: np.ndarray,
    contact_point: np.ndarray,
    newtons_per_meter: float,
    radius: float = 0.001,
    rgba=Rgba(1.0, 0.0, 0.0, 1.0),
) -> None:
    """A point contact arrow represents equal and opposite forces from the contact point."""

    # Create arrow
    height = np.linalg.norm(force) / newtons_per_meter
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


def add_hydroelastic_arrow(
    meshcat: Meshcat,
    path: str,
    force: np.ndarray,
    centroid: np.ndarray,
    newtons_per_meter: float,
    radius: float = 0.001,
    rgba=Rgba(1.0, 0.0, 0.0, 1.0),
) -> None:
    """A hydroelastic arrow represents a single force from the centroid (not equal and opposite)."""

    # Create arrow
    height = np.linalg.norm(force) / newtons_per_meter
    cylinder = Cylinder(radius, height)
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

    # Transform arrow
    meshcat.SetTransform(
        path, RigidTransform(RotationMatrix.MakeFromOneVector(force, 2), centroid)
    )  # Arrow starts along z-axis (axis 2)
    meshcat.SetTransform(
        path + "/cylinder", RigidTransform([0.0, 0.0, height / 2.0])
    )  # Arrow starts at centroid and goes into single direction
    meshcat.SetTransform(
        path + "/head", RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.0, 0.0, height + arrowhead_height])
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
    arg_parser.add_argument(
        "--newtons_per_meter",
        default=1e2,
        type=float,
        help="How many meters the force arrows should be long for each Newton of force.",
    )
    args = arg_parser.parse_args()
    # TODO: Optionally saving html

    assert args.manipuland in ["outer", "inner", "both", "none"]

    log_dir = os.path.join(args.data, "time_logs")
    times = np.loadtxt(os.path.join(log_dir, "outer_manipuland_pose_times.txt"))
    (
        outer_hydroelastic_centroids,
        outer_hydroelastic_contact_forces,
        outer_point_contact_points,
        outer_point_contact_forces,
        outer_manipuland_poses,
    ) = load_data(log_dir, outer=True)
    (
        inner_hydroelastic_centroids,
        inner_hydroelastic_contact_forces,
        inner_point_contact_points,
        inner_point_contact_forces,
        inner_manipuland_poses,
    ) = load_data(log_dir, outer=False)

    time_idx = np.abs(times - args.time).argmin()
    time_diff = abs(times[time_idx] - args.time)
    if time_diff > 0.01:
        print(f"Demanded and available time differ by {time_diff}s. Showing forces for time {times[time_idx]}s.")

    outer_hydroelastic_centroid = outer_hydroelastic_centroids[time_idx]
    outer_hydroelastic_contact_force = outer_hydroelastic_contact_forces[time_idx]
    outer_point_contact_point = outer_point_contact_points[time_idx]
    outer_point_contact_force = outer_point_contact_forces[time_idx]
    outer_manipuland_pose = outer_manipuland_poses[time_idx]
    inner_hydroelastic_centroid = inner_hydroelastic_centroids[time_idx]
    inner_hydroelastic_contact_force = inner_hydroelastic_contact_forces[time_idx]
    inner_point_contact_point = inner_point_contact_points[time_idx]
    inner_point_contact_force = inner_point_contact_forces[time_idx]
    inner_manipuland_pose = inner_manipuland_poses[time_idx]

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = get_parser(plant)

    # Visualize manipulands
    experiment_description_path = os.path.join(args.data, "experiment_description.yaml")
    experiment_description = yaml.safe_load(open(experiment_description_path, "r"))
    manipuland_base_link_name = experiment_description["script"]["args"]["manipuland_base_link_name"]
    if args.manipuland in ["outer", "both"]:
        outer_manipuland_directive_path = os.path.join(
            args.data, experiment_description["script"]["args"]["manipuland_directive"]
        )
        outer_manipuland_directive = LoadModelDirectives(outer_manipuland_directive_path)
        ProcessModelDirectives(outer_manipuland_directive, parser)
        quat = outer_manipuland_pose[:4]
        quat_normalized = quat / np.linalg.norm(quat)
        plant.SetDefaultFreeBodyPose(
            plant.GetBodyByName(manipuland_base_link_name),
            RigidTransform(Quaternion(quat_normalized), outer_manipuland_pose[4:]),
        )

    if args.manipuland in ["inner", "both"]:
        inner_manipuland_sdf_path = os.path.join(args.data, "meshes", "processed_mesh.sdf")
        inner_manipuland = parser.AddModelFromFile(inner_manipuland_sdf_path, "inner_manipuland")
        quat = outer_manipuland_pose[:4]
        quat_normalized = quat / np.linalg.norm(quat)
        plant.SetDefaultFreeBodyPose(
            plant.GetBodyByName(manipuland_base_link_name, inner_manipuland),
            RigidTransform(Quaternion(quat_normalized), inner_manipuland_pose[4:]),
        )

    meshcat = StartMeshcat()
    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.role = Role.kProximity
    _ = MeshcatVisualizer.AddToBuilder(builder, scene_graph.get_query_output_port(), meshcat, meshcat_params)

    if args.hydroelastic:
        # TODO: Also visualize contact torques (see C++ visualizer)
        for i, (force, centroid) in enumerate(zip(outer_hydroelastic_contact_force, outer_hydroelastic_centroid)):
            if np.linalg.norm(force) > 0.0:
                add_hydroelastic_arrow(
                    meshcat,
                    path=f"contact_forces/outer_sim/force_{i}",
                    force=force,
                    centroid=centroid,
                    newtons_per_meter=args.newtons_per_meter,
                    rgba=Rgba(0.0, 1.0, 0.0, 1.0),
                )

        for i, (force, centroid) in enumerate(zip(inner_hydroelastic_contact_force, inner_hydroelastic_centroid)):
            if np.linalg.norm(force) > 0.0:
                add_hydroelastic_arrow(
                    meshcat,
                    path=f"contact_forces/inner_sim/force_{i}",
                    force=force,
                    centroid=centroid,
                    newtons_per_meter=args.newtons_per_meter,
                    rgba=Rgba(1.0, 0.0, 0.0, 1.0),
                )

    else:
        for i, (force, point) in enumerate(zip(outer_point_contact_force, outer_point_contact_point)):
            if np.linalg.norm(force) > 0.0:
                add_point_contact_arrow(
                    meshcat,
                    path=f"contact_forces/outer_sim/force_{i}",
                    force=force,
                    contact_point=point,
                    newtons_per_meter=args.newtons_per_meter,
                    rgba=Rgba(0.0, 1.0, 0.0, 1.0),
                )

        for i, (force, point) in enumerate(zip(inner_point_contact_force, inner_point_contact_point)):
            if np.linalg.norm(force) > 0.0:
                add_point_contact_arrow(
                    meshcat,
                    path=f"contact_forces/inner_sim/force_{i}",
                    force=force,
                    contact_point=point,
                    newtons_per_meter=args.newtons_per_meter,
                    rgba=Rgba(1.0, 0.0, 0.0, 1.0),
                )

    # inner_meshcat.Delete("contacts")

    plant.Finalize()
    diagram = builder.Build()

    # Need to simulate for visualization to work
    simulator = Simulator(diagram)
    simulator.AdvanceTo(0.0)
    print("Finished loading visualization")

    # Sleep to give user enough time to click on meshcat link
    time.sleep(5.0)


if __name__ == "__main__":
    main()
