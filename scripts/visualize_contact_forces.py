"""Script for visualizing a collision representation and its contact forces."""

import argparse
import os
import yaml
import time

import numpy as np
import open3d as o3d
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

from sim2sim.util import get_parser, vector_pose_to_rigidtransform, get_principal_component

# Meshcat item names
TIME_SLIDER_NAME = "time"
TOGGLE_INNER_FORCES_BUTTON_NAME = "Toggle inner forces default visibility"
TOGGLE_OUTER_FORCES_BUTTON_NAME = "Toggle outer forces default visibility"
TOGGLE_INNER_GENERALIZED_FORCES_BUTTON_NAME = "Toggle inner generalized forces default visibility"
TOGGLE_OUTER_GENERALIZED_FORCES_BUTTON_NAME = "Toggle outer generalized forces default visibility"
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

    # Generalized contact forces
    generalized_contact_forces = np.loadtxt(os.path.join(log_dir, f"{prefix}_manipuland_contact_forces.txt"))

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
        generalized_contact_forces,
        hydroelastic_centroids,
        hydroelastic_contact_forces,
        hydroelastic_contact_torques,
        point_contact_points,
        point_contact_forces,
        manipuland_poses,
    )


def add_equal_opposite_force_arrow(
    meshcat: Meshcat,
    path: str,
    force: np.ndarray,
    contact_point: np.ndarray,
    newtons_per_meter: float,
    radius: float = 0.001,
    rgba: Rgba = Rgba(1.0, 0.0, 0.0, 1.0),
) -> None:
    """
    A contact force arrow that represents equal and opposite forces from the contact point.
    Example: Point contact result forces.
    """

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


def add_single_direction_force_arrow(
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
    """
    A contact force arrow that represents a single force from the centroid (not equal and opposite).
    Examples: Generalized contact forces, hydroelastic contact result forces.
    """

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


def get_separation_vec(outer_translations: np.ndarray, inner_translations: np.ndarray, viz: bool = False) -> np.ndarray:
    """
    Returns the vector perpendicular to both the manipuland translations and the z-axis where the translations are the
    combined outer and inner translations.

    :param outer_translations: The outer translations of shape (N,3).
    :param inner_translations: The inner translations of shape (N,3).
    :param viz: Whether to visualize the translation points with the principal component and separations vectors.
    :return: The separation vector of shape (3,).
    """
    combined_translations = np.concatenate([outer_translations, inner_translations], axis=0)
    principle_component = get_principal_component(combined_translations)
    z_axis = [0.0, 0.0, 1.0]
    separation_vec = np.cross(principle_component, z_axis)

    if viz:
        # Visualize the outer translations in green, the inner translations in orange, the principal component in blue,
        # and the separation vector in red
        outer_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(outer_translations))
        outer_pcd.paint_uniform_color([0.0, 1.0, 0.0])
        inner_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inner_translations))
        inner_pcd.paint_uniform_color([1.0, 0.5, 0.0])  # orange
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(
            np.array([-principle_component, principle_component, -separation_vec, separation_vec])
        )
        lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3]]))
        lines.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        o3d.visualization.draw_geometries([outer_pcd, inner_pcd, lines, world_frame])

    return separation_vec


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
    parser.add_argument(
        "--separation_distance",
        default=0.0,
        type=float,
        help="The distance in meters that the outer and inner manipuland should be separated from each other. "
        + "This only has an effect if `--manipuland` is 'both'.",
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
        outer_generalized_contact_forces,
        outer_hydroelastic_centroids,
        outer_hydroelastic_contact_forces,
        outer_hydroelastic_contact_torques,
        outer_point_contact_points,
        outer_point_contact_forces,
        outer_manipuland_poses,
    ) = load_data(log_dir, outer=True)
    (
        inner_generalized_contact_forces,
        inner_hydroelastic_centroids,
        inner_hydroelastic_contact_forces,
        inner_hydroelastic_contact_torques,
        inner_point_contact_points,
        inner_point_contact_forces,
        inner_manipuland_poses,
    ) = load_data(log_dir, outer=False)

    if args.manipuland == "both" and args.separation_distance > 0.0:
        separation_direction_vec = get_separation_vec(outer_manipuland_poses[:, 4:], inner_manipuland_poses[:, 4:])
        separation_direction_vec_unit = separation_direction_vec / np.linalg.norm(separation_direction_vec)
        separation_vec = args.separation_distance / 2.0 * separation_direction_vec_unit

        add_force_vec = lambda a, vec: np.array([el + vec if np.linalg.norm(el) > 0.0 else el for el in a])

        outer_hydroelastic_centroids = add_force_vec(outer_hydroelastic_centroids, separation_vec)
        outer_point_contact_points = add_force_vec(outer_point_contact_points, separation_vec)
        outer_manipuland_poses[:, 4:] += separation_vec

        inner_hydroelastic_centroids = add_force_vec(inner_hydroelastic_centroids, -separation_vec)
        inner_point_contact_points = add_force_vec(inner_point_contact_points, -separation_vec)
        inner_manipuland_poses[:, 4:] -= separation_vec

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
    toggle_inner_generalized_forces_button_clicks = 0
    inner_generalized_forces_visible = True
    meshcat.AddButton(name=TOGGLE_INNER_GENERALIZED_FORCES_BUTTON_NAME)
    toggle_outer_generalized_forces_button_clicks = 0
    outer_generalized_forces_visible = True
    meshcat.AddButton(name=TOGGLE_OUTER_GENERALIZED_FORCES_BUTTON_NAME)
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
        if (
            meshcat.GetButtonClicks(TOGGLE_INNER_GENERALIZED_FORCES_BUTTON_NAME)
            > toggle_inner_generalized_forces_button_clicks
        ):
            toggle_inner_generalized_forces_button_clicks += 1
            inner_generalized_forces_visible = not inner_generalized_forces_visible
        meshcat.SetProperty("contact_forces/inner_sim_generalized", "visible", inner_generalized_forces_visible)
        if (
            meshcat.GetButtonClicks(TOGGLE_OUTER_GENERALIZED_FORCES_BUTTON_NAME)
            > toggle_outer_generalized_forces_button_clicks
        ):
            toggle_outer_generalized_forces_button_clicks += 1
            outer_generalized_forces_visible = not outer_generalized_forces_visible
        meshcat.SetProperty("contact_forces/outer_sim_generalized", "visible", outer_generalized_forces_visible)
        if meshcat.GetButtonClicks(TOGGLE_INNER_MANIPULAND_BUTTON_NAME) > toggle_inner_manipuland_button_clicks:
            toggle_inner_manipuland_button_clicks += 1
            inner_manipuland_visible = not inner_manipuland_visible
        meshcat.SetProperty(f"visualizer/inner_manipuland", "visible", inner_manipuland_visible)
        if meshcat.GetButtonClicks(TOGGLE_OUTER_MANIPULAND_BUTTON_NAME) > toggle_outer_manipuland_button_clicks:
            toggle_outer_manipuland_button_clicks += 1
            outer_manipuland_visible = not outer_manipuland_visible
        meshcat.SetProperty(f"visualizer/{manipuland_name}", "visible", outer_manipuland_visible)

        # Visualize generalized contact forces
        outer_generalized_contact_force = outer_generalized_contact_forces[time_idx]
        if np.linalg.norm(outer_generalized_contact_force) > 0.0:
            add_single_direction_force_arrow(
                meshcat,
                path="contact_forces/outer_sim_generalized",
                force=outer_generalized_contact_force[3:],
                torque=outer_generalized_contact_force[:3],
                centroid=outer_manipuland_poses[time_idx][4:],
                newtons_per_meter=args.newtons_per_meter,
                newton_meters_per_meter=args.newton_meters_per_meter,
                force_rgba=Rgba(0.0, 0.0, 1.0, 1.0),  # blue
                torque_rgba=Rgba(0.6, 0.6, 1.0, 1.0),  # purple
            )
        inner_generalized_contact_force = inner_generalized_contact_forces[time_idx]
        if np.linalg.norm(inner_generalized_contact_force) > 0.0:
            add_single_direction_force_arrow(
                meshcat,
                path="contact_forces/inner_sim_generalized",
                force=inner_generalized_contact_force[3:],
                torque=inner_generalized_contact_force[:3],
                centroid=inner_manipuland_poses[time_idx][4:],
                newtons_per_meter=args.newtons_per_meter,
                newton_meters_per_meter=args.newton_meters_per_meter,
                force_rgba=Rgba(1.0, 0.0, 0.5, 1.0),  # pink
                torque_rgba=Rgba(1.0, 0.6, 0.8, 1.0),  # light pink
            )

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
                    add_single_direction_force_arrow(
                        meshcat,
                        path=f"contact_forces/outer_sim/force_{i}",
                        force=force,
                        torque=torque,
                        centroid=centroid,
                        newtons_per_meter=args.newtons_per_meter,
                        newton_meters_per_meter=args.newton_meters_per_meter,
                        force_rgba=Rgba(0.0, 1.0, 0.0, 1.0),
                        torque_rgba=Rgba(0.0, 1.0, 1.0, 1.0),  # light blue
                    )

            for i, (force, torque, centroid) in enumerate(
                zip(
                    inner_hydroelastic_contact_forces[time_idx],
                    inner_hydroelastic_contact_torques[time_idx],
                    inner_hydroelastic_centroids[time_idx],
                )
            ):
                if np.linalg.norm(force) > 0.0:
                    add_single_direction_force_arrow(
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
                    add_equal_opposite_force_arrow(
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
                    add_equal_opposite_force_arrow(
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
