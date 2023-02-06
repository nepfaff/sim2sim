"""Script for visualizing a collision representation and its contact forces."""

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

from sim2sim.util import get_parser, vector_pose_to_rigidtransform

# Meshcat item names
TIME_SLIDER_NAME = "time"
TOGGLE_INNER_FORCES_BUTTON_NAME = "Toggle inner forces default visibility"
TOGGLE_OUTER_FORCES_BUTTON_NAME = "Toggle outer forces default visibility"
TOGGLE_INNER_MANIPULAND_BUTTON_NAME = "Toggle inner manipuland default visibility"
TOGGLE_OUTER_MANIPULAND_BUTTON_NAME = "Toggle outer manipuland default visibility"


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
    contact_result_torques_raw = np.load(
        os.path.join(log_dir, f"{prefix}_hydroelastic_contact_result_torques.npy"),
        allow_pickle=True,
    )
    hydroelastic_contact_torques = process_array(contact_result_torques_raw)

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
        hydroelastic_contact_torques,
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
    rgba: Rgba = Rgba(1.0, 0.0, 0.0, 1.0),
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
    torque: np.ndarray,
    centroid: np.ndarray,
    newtons_per_meter: float,
    newton_meters_per_meter: float,
    radius: float = 0.001,
    force_rgba: Rgba = Rgba(1.0, 0.0, 0.0, 1.0),
    torque_rgba: Rgba = Rgba(0.0, 0.0, 1.0, 1.0),
) -> None:
    """A hydroelastic arrow represents a single force from the centroid (not equal and opposite)."""

    # Create force arrow
    force_height = np.linalg.norm(force) / newtons_per_meter
    force_cylinder = Cylinder(radius, force_height)
    meshcat.SetObject(
        path=path + "/force/cylinder",
        shape=force_cylinder,
        rgba=force_rgba,
    )
    arrowhead_height = arrowhead_width = radius * 2.0
    arrowhead = MeshcatCone(arrowhead_height, arrowhead_width, arrowhead_width)
    meshcat.SetObject(
        path=path + "/force/head",
        shape=arrowhead,
        rgba=force_rgba,
    )

    # Create torque arrow
    torque_height = np.linalg.norm(torque) / newton_meters_per_meter
    torque_cylinder = Cylinder(radius, torque_height)
    meshcat.SetObject(
        path=path + "/torque/cylinder",
        shape=torque_cylinder,
        rgba=torque_rgba,
    )
    meshcat.SetObject(
        path=path + "/torque/head",
        shape=arrowhead,
        rgba=torque_rgba,
    )

    # Transform force arrow
    meshcat.SetTransform(
        path + "/force", RigidTransform(RotationMatrix.MakeFromOneVector(force, 2), centroid)
    )  # Arrow starts along z-axis (axis 2)
    meshcat.SetTransform(
        path + "/force/cylinder", RigidTransform([0.0, 0.0, force_height / 2.0])
    )  # Arrow starts at centroid and goes into single direction
    meshcat.SetTransform(
        path + "/force/head",
        RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.0, 0.0, force_height + arrowhead_height]),
    )

    # Transform torque arrow
    meshcat.SetTransform(
        path + "/torque", RigidTransform(RotationMatrix.MakeFromOneVector(torque, 2), centroid)
    )  # Arrow starts along z-axis (axis 2)
    meshcat.SetTransform(
        path + "/torque/cylinder", RigidTransform([0.0, 0.0, torque_height / 2.0])
    )  # Arrow starts at centroid and goes into single direction
    meshcat.SetTransform(
        path + "/torque/head",
        RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.0, 0.0, torque_height + arrowhead_height]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to the experiment data folder.",
    )
    parser.add_argument(
        "--hydroelastic", action="store_true", help="Whether to plot hydroelastic or point contact forces."
    )
    parser.add_argument(
        "--manipuland",
        default="both",
        type=str,
        help="The manipuland to visualize. Options are 'outer', 'inner', 'both', and 'none'.",
    )
    parser.add_argument(
        "--newtons_per_meter",
        default=1e2,
        type=float,
        help="HSets the length scale of the force vectors.",
    )
    parser.add_argument(
        "--newton_meters_per_meter",
        default=1.0,
        type=float,
        help="Sets the length scale of the torque/ moment vectors.",
    )
    parser.add_argument("--save_html", action="store_true", help="Whether to save the meshcat HTML.")

    args = parser.parse_args()
    assert args.manipuland in ["outer", "inner", "both", "none"]

    return args


def main():
    args = parse_args()

    # Load data
    log_dir = os.path.join(args.data, "time_logs")
    times = np.loadtxt(os.path.join(log_dir, "outer_manipuland_pose_times.txt"))
    (
        outer_hydroelastic_centroids,
        outer_hydroelastic_contact_forces,
        outer_hydroelastic_contact_torques,
        outer_point_contact_points,
        outer_point_contact_forces,
        outer_manipuland_poses,
    ) = load_data(log_dir, outer=True)
    (
        inner_hydroelastic_centroids,
        inner_hydroelastic_contact_forces,
        inner_hydroelastic_contact_torques,
        inner_point_contact_points,
        inner_point_contact_forces,
        inner_manipuland_poses,
    ) = load_data(log_dir, outer=False)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = get_parser(plant)

    # Visualize manipulands
    experiment_description_path = os.path.join(args.data, "experiment_description.yaml")
    experiment_description = yaml.safe_load(open(experiment_description_path, "r"))
    manipuland_name = experiment_description["logger"]["args"]["manipuland_name"]
    manipuland_base_link_name = experiment_description["logger"]["args"]["manipuland_base_link_name"]
    if args.manipuland in ["outer", "both"]:
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

    plant.Finalize()
    diagram = builder.Build()

    # Need to simulate for visualization to work
    simulator = Simulator(diagram)
    simulator.AdvanceTo(0.0)

    # Add slider for stepping through time
    meshcat.AddSlider(
        name=TIME_SLIDER_NAME,
        min=0.0,
        max=times[-1],
        step=times[1] - times[0],
        value=0.0,
        decrement_keycode="ArrowLeft",
        increment_keycode="ArrowRight",
    )

    # Add buttons for setting item visibility
    toggle_inner_forces_button_clicks = 0
    inner_forces_visible = True
    meshcat.AddButton(name=TOGGLE_INNER_FORCES_BUTTON_NAME)
    toggle_outer_forces_button_clicks = 0
    outer_forces_visible = True
    meshcat.AddButton(name=TOGGLE_OUTER_FORCES_BUTTON_NAME)
    toggle_inner_manipuland_button_clicks = 0
    inner_manipuland_visible = True
    meshcat.AddButton(name=TOGGLE_INNER_MANIPULAND_BUTTON_NAME)
    toggle_outer_manipuland_button_clicks = 0
    outer_manipuland_visible = True
    meshcat.AddButton(name=TOGGLE_OUTER_MANIPULAND_BUTTON_NAME)

    print("Entering infinite loop. Force quit to exit.")  # TODO: Check for user input in non-blocking way
    current_time = -1.0  # Force initial meshcat update
    while True:
        new_demanded_time = meshcat.GetSliderValue(TIME_SLIDER_NAME)
        time_idx = np.abs(times - new_demanded_time).argmin()
        new_time = times[time_idx]

        # Only update meshcat if the data changed
        if new_time == current_time:
            time.sleep(0.1)
            continue
        else:
            current_time = new_time

        # Delete all force arrows before adding the new ones
        meshcat.Delete("contact_forces")

        # Update item visibility
        if meshcat.GetButtonClicks(TOGGLE_INNER_FORCES_BUTTON_NAME) > toggle_inner_forces_button_clicks:
            toggle_inner_forces_button_clicks += 1
            inner_forces_visible = not inner_forces_visible
        meshcat.SetProperty("contact_forces/inner_sim", "visible", inner_forces_visible)
        if meshcat.GetButtonClicks(TOGGLE_OUTER_FORCES_BUTTON_NAME) > toggle_outer_forces_button_clicks:
            toggle_outer_forces_button_clicks += 1
            outer_forces_visible = not outer_forces_visible
        meshcat.SetProperty("contact_forces/outer_sim", "visible", outer_forces_visible)
        if meshcat.GetButtonClicks(TOGGLE_INNER_MANIPULAND_BUTTON_NAME) > toggle_inner_manipuland_button_clicks:
            toggle_inner_manipuland_button_clicks += 1
            inner_manipuland_visible = not inner_manipuland_visible
        meshcat.SetProperty(f"visualizer/inner_manipuland", "visible", inner_manipuland_visible)
        if meshcat.GetButtonClicks(TOGGLE_OUTER_MANIPULAND_BUTTON_NAME) > toggle_outer_manipuland_button_clicks:
            toggle_outer_manipuland_button_clicks += 1
            outer_manipuland_visible = not outer_manipuland_visible
        meshcat.SetProperty(f"visualizer/{manipuland_name}", "visible", outer_manipuland_visible)

        # Visualize new contact forces
        if args.hydroelastic:
            # NOTE: The hydroelastic forces seem very different to the ones in the recorded HTMLs of the simulation.
            # Further investigation is needed to determine why this is the case. For now, it is better to use this visualizer
            # for point contact visualizations.
            for i, (force, torque, centroid) in enumerate(
                zip(
                    outer_hydroelastic_contact_forces[time_idx],
                    outer_hydroelastic_contact_torques[time_idx],
                    outer_hydroelastic_centroids[time_idx],
                )
            ):
                if np.linalg.norm(force) > 0.0:
                    add_hydroelastic_arrow(
                        meshcat,
                        path=f"contact_forces/outer_sim/force_{i}",
                        force=force,
                        torque=torque,
                        centroid=centroid,
                        newtons_per_meter=args.newtons_per_meter,
                        newton_meters_per_meter=args.newton_meters_per_meter,
                        force_rgba=Rgba(0.0, 1.0, 0.0, 1.0),
                        torque_rgba=Rgba(0.0, 1.0, 1.0, 1.0),  # blue
                    )

            for i, (force, torque, centroid) in enumerate(
                zip(
                    inner_hydroelastic_contact_forces[time_idx],
                    inner_hydroelastic_contact_torques[time_idx],
                    inner_hydroelastic_centroids[time_idx],
                )
            ):
                if np.linalg.norm(force) > 0.0:
                    add_hydroelastic_arrow(
                        meshcat,
                        path=f"contact_forces/inner_sim/force_{i}",
                        force=force,
                        torque=torque,
                        centroid=centroid,
                        newtons_per_meter=args.newtons_per_meter,
                        newton_meters_per_meter=args.newton_meters_per_meter,
                        force_rgba=Rgba(1.0, 0.0, 0.0, 1.0),
                        torque_rgba=Rgba(1.0, 0.5, 0.0, 1.0),  # orange
                    )
        else:
            for i, (force, point) in enumerate(
                zip(outer_point_contact_forces[time_idx], outer_point_contact_points[time_idx])
            ):
                if np.linalg.norm(force) > 0.0:
                    add_point_contact_arrow(
                        meshcat,
                        path=f"contact_forces/outer_sim/force_{i}",
                        force=force,
                        contact_point=point,
                        newtons_per_meter=args.newtons_per_meter,
                        rgba=Rgba(0.0, 1.0, 0.0, 1.0),
                    )

            for i, (force, point) in enumerate(
                zip(inner_point_contact_forces[time_idx], inner_point_contact_points[time_idx])
            ):
                if np.linalg.norm(force) > 0.0:
                    add_point_contact_arrow(
                        meshcat,
                        path=f"contact_forces/inner_sim/force_{i}",
                        force=force,
                        contact_point=point,
                        newtons_per_meter=args.newtons_per_meter,
                        rgba=Rgba(1.0, 0.0, 0.0, 1.0),
                    )

        # Update manipuland mesh poses
        meshcat.SetTransform(
            f"visualizer/{manipuland_name}/{manipuland_base_link_name}",
            vector_pose_to_rigidtransform(outer_manipuland_poses[time_idx]),
        )
        meshcat.SetTransform(
            f"visualizer/inner_manipuland/{manipuland_base_link_name}",
            vector_pose_to_rigidtransform(inner_manipuland_poses[time_idx]),
        )

        if args.save_html:
            # Overwrite saved HTML
            html = meshcat.StaticHtml()
            html_path = os.path.join(args.data, "contact_force_visualizer.html")
            with open(html_path, "w") as f:
                f.write(html)


if __name__ == "__main__":
    main()
