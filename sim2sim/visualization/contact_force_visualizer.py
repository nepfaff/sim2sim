"""Script for visualizing a collision representation and its contact forces."""

import os
import yaml
import time
from typing import Tuple

import numpy as np
import open3d as o3d
from pydrake.all import (
    StartMeshcat,
    MeshcatVisualizerParams,
    Role,
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

# Minimum force/ torque magnitude for Drake operations to work
FORCE_TORQUE_MIN_MAGNITUDE = 1e-10


class ContactForceVisualizer:
    """A visualizer for simultaneously visualizing both outer and inner manipulands with their contact forces."""

    def __init__(
        self,
        data_path: str,
        manipuland: str,
        separation_distance: float,
        save_html: bool,
        newtons_per_meter: float,
        newton_meters_per_meter: float,
        hydroelastic: bool,
        force_magnitude_theshold: float = 1e-9,
    ):
        """
        :param data_path: Path to the experiment data folder.
        :param manipuland: The manipuland to visualize. Options are 'outer', 'inner', 'both', and 'none'.
        :param separation_distance: The distance in meters that the outer and inner manipuland should be separated from
            each other. This only has an effect if `--manipuland` is 'both'.
        :param save_html: Whether to save the meshcat HTML.
        :param newtons_per_meter: Sets the length scale of the force vectors.
        :param newton_meters_per_meter: Sets the length scale of the torque/ moment vectors.
        :param hydroelastic: Whether to plot hydroelastic or point contact forces.
        :param force_magnitude_theshold: Don't visualize forces that have a magnitude of less than this.
        """
        assert manipuland in ["outer", "inner", "both", "none"]

        self._data_path = data_path
        self._manipuland = manipuland
        self._separation_distance = separation_distance
        self._save_html = save_html
        self._newtons_per_meter = newtons_per_meter
        self._newton_meters_per_meter = newton_meters_per_meter
        self._hydroelastic = hydroelastic
        self._force_magnitude_theshold = force_magnitude_theshold

        self._log_dir = os.path.join(data_path, "time_logs")
        self._is_setup = False

        self._meshcat = StartMeshcat()

        # Meshcat button data
        self._toggle_inner_forces_button_clicks = 0
        self._inner_forces_visible = True
        self._toggle_outer_forces_button_clicks = 0
        self._outer_forces_visible = True
        self._toggle_inner_generalized_forces_button_clicks = 0
        self._inner_generalized_forces_visible = True
        self._toggle_outer_generalized_forces_button_clicks = 0
        self._outer_generalized_forces_visible = True
        self._toggle_inner_manipuland_button_clicks = 0
        self._inner_manipuland_visible = True
        self._toggle_outer_manipuland_button_clicks = 0
        self._outer_manipuland_visible = True

        # Create Drake environment for visualizing SDF files
        self._builder = DiagramBuilder()
        self._plant, self._scene_graph = AddMultibodyPlantSceneGraph(self._builder, 1e-3)
        self._parser = get_parser(self._plant)

        # Extract params from the experiment description
        experiment_description_path = os.path.join(self._data_path, "experiment_description.yaml")
        self._experiment_description = yaml.safe_load(open(experiment_description_path, "r"))
        self._manipuland_name = self._experiment_description["logger"]["args"]["manipuland_name"]
        self._manipuland_base_link_name = self._experiment_description["logger"]["args"]["manipuland_base_link_name"]
        self._is_pipeline_comparison = self._experiment_description["script"]["args"]["is_pipeline_comparison"]

        # Load data
        self._times = np.loadtxt(os.path.join(self._log_dir, "outer_manipuland_pose_times.txt"))
        (
            self._outer_generalized_contact_forces,
            self._outer_hydroelastic_centroids,
            self._outer_hydroelastic_contact_forces,
            self._outer_hydroelastic_contact_torques,
            self._outer_point_contact_points,
            self._outer_point_contact_forces,
            self._outer_manipuland_poses,
        ) = self._load_data(self._log_dir, outer=True)
        (
            self._inner_generalized_contact_forces,
            self._inner_hydroelastic_centroids,
            self._inner_hydroelastic_contact_forces,
            self._inner_hydroelastic_contact_torques,
            self._inner_point_contact_points,
            self._inner_point_contact_forces,
            self._inner_manipuland_poses,
        ) = self._load_data(self._log_dir, outer=False)

        if self._manipuland == "both" and self._separation_distance > 0.0:
            separation_direction_vec = self._get_separation_direction_vec(
                self._outer_manipuland_poses[:, 4:], self._inner_manipuland_poses[:, 4:]
            )
            separation_direction_vec_unit = separation_direction_vec / np.linalg.norm(separation_direction_vec)
            self._separation_vec = self._separation_distance / 2.0 * separation_direction_vec_unit

            self._modify_data_for_side_by_side_visualization()

    def _modify_data_for_side_by_side_visualization(self) -> None:
        add_force_vec = lambda a, vec: np.array([el + vec if np.linalg.norm(el) > 0.0 else el for el in a])

        self._outer_hydroelastic_centroids = add_force_vec(self._outer_hydroelastic_centroids, self._separation_vec)
        self._outer_point_contact_points = add_force_vec(self._outer_point_contact_points, self._separation_vec)
        self._outer_manipuland_poses[:, 4:] += self._separation_vec

        self._inner_hydroelastic_centroids = add_force_vec(self._inner_hydroelastic_centroids, -self._separation_vec)
        self._inner_point_contact_points = add_force_vec(self._inner_point_contact_points, -self._separation_vec)
        self._inner_manipuland_poses[:, 4:] -= self._separation_vec

    def _visualize_manipulands(self) -> None:
        """Visualizes the manipuland(s) at the world origin."""
        if self._manipuland in ["outer", "both"]:
            if self._is_pipeline_comparison:
                outer_manipuland_sdf_path = os.path.join(self._data_path, "meshes", f"outer_processed_mesh.sdf")
                self._parser.AddModelFromFile(outer_manipuland_sdf_path, self._manipuland_name)
            else:
                outer_manipuland_directive_path = os.path.join(
                    self._data_path, self._experiment_description["script"]["args"]["manipuland_directive"]
                )
                outer_manipuland_directive = LoadModelDirectives(outer_manipuland_directive_path)
                ProcessModelDirectives(outer_manipuland_directive, self._parser)
        if self._manipuland in ["inner", "both"]:
            inner_manipuland_sdf_path = os.path.join(
                self._data_path, "meshes", f"{'inner_' if self._is_pipeline_comparison else ''}processed_mesh.sdf"
            )
            self._parser.AddModelFromFile(inner_manipuland_sdf_path, "inner_manipuland")

    @staticmethod
    def _load_data(log_dir: str, outer: bool):
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

    def _add_equal_opposite_force_arrow(
        self,
        path: str,
        force: np.ndarray,
        contact_point: np.ndarray,
        radius: float = 0.001,
        rgba: Rgba = Rgba(1.0, 0.0, 0.0, 1.0),
    ) -> None:
        """
        A contact force arrow that represents equal and opposite forces from the contact point.
        Example: Point contact result forces.
        """
        force_magnitude = np.linalg.norm(force)
        if force_magnitude < FORCE_TORQUE_MIN_MAGNITUDE:
            return

        # Create arrow
        height = force_magnitude / self._newtons_per_meter
        # Cylinder gets scaled to twice the contact force length because we draw both (equal and opposite) forces
        cylinder = Cylinder(radius, 2 * height)
        self._meshcat.SetObject(
            path=path + "/cylinder",
            shape=cylinder,
            rgba=rgba,
        )
        arrowhead_height = arrowhead_width = radius * 2.0
        arrowhead = MeshcatCone(arrowhead_height, arrowhead_width, arrowhead_width)
        self._meshcat.SetObject(
            path=path + "/head",
            shape=arrowhead,
            rgba=rgba,
        )
        self._meshcat.SetObject(
            path=path + "/tail",
            shape=arrowhead,
            rgba=rgba,
        )

        # Transform arrow
        self._meshcat.SetTransform(
            path, RigidTransform(RotationMatrix.MakeFromOneVector(force, 2), contact_point)
        )  # Arrow starts along z-axis (axis 2)
        self._meshcat.SetTransform(path + "/head", RigidTransform([0.0, 0.0, -height - arrowhead_height]))
        self._meshcat.SetTransform(
            path + "/tail", RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.0, 0.0, height + arrowhead_height])
        )

    def _add_single_direction_force_arrow(
        self,
        path: str,
        force: np.ndarray,
        torque: np.ndarray,
        centroid: np.ndarray,
        radius: float = 0.001,
        force_rgba: Rgba = Rgba(1.0, 0.0, 0.0, 1.0),
        torque_rgba: Rgba = Rgba(0.0, 0.0, 1.0, 1.0),
    ) -> None:
        """
        A contact force arrow that represents a single force from the centroid (not equal and opposite).
        Examples: Generalized contact forces, hydroelastic contact result forces.
        """
        arrowhead_height = arrowhead_width = radius * 2.0
        arrowhead = MeshcatCone(arrowhead_height, arrowhead_width, arrowhead_width)

        force_magnitude = np.linalg.norm(force)

        # Create force arrow
        if force_magnitude > FORCE_TORQUE_MIN_MAGNITUDE:
            force_height = np.linalg.norm(force) / self._newtons_per_meter
            force_cylinder = Cylinder(radius, force_height)
            self._meshcat.SetObject(
                path=path + "/force/cylinder",
                shape=force_cylinder,
                rgba=force_rgba,
            )
            self._meshcat.SetObject(
                path=path + "/force/head",
                shape=arrowhead,
                rgba=force_rgba,
            )

        torque_magnitude = np.linalg.norm(torque)

        # Create torque arrow
        if torque_magnitude > FORCE_TORQUE_MIN_MAGNITUDE:
            torque_height = torque_magnitude / self._newton_meters_per_meter
            torque_cylinder = Cylinder(radius, torque_height)
            self._meshcat.SetObject(
                path=path + "/torque/cylinder",
                shape=torque_cylinder,
                rgba=torque_rgba,
            )
            self._meshcat.SetObject(
                path=path + "/torque/head",
                shape=arrowhead,
                rgba=torque_rgba,
            )

        # Transform force arrow
        if force_magnitude > FORCE_TORQUE_MIN_MAGNITUDE:
            self._meshcat.SetTransform(
                path + "/force", RigidTransform(RotationMatrix.MakeFromOneVector(force, 2), centroid)
            )  # Arrow starts along z-axis (axis 2)
            self._meshcat.SetTransform(
                path + "/force/cylinder", RigidTransform([0.0, 0.0, force_height / 2.0])
            )  # Arrow starts at centroid and goes into single direction
            self._meshcat.SetTransform(
                path + "/force/head",
                RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.0, 0.0, force_height + arrowhead_height]),
            )

        # Transform torque arrow
        if torque_magnitude > FORCE_TORQUE_MIN_MAGNITUDE:
            self._meshcat.SetTransform(
                path + "/torque", RigidTransform(RotationMatrix.MakeFromOneVector(torque, 2), centroid)
            )  # Arrow starts along z-axis (axis 2)
            self._meshcat.SetTransform(
                path + "/torque/cylinder", RigidTransform([0.0, 0.0, torque_height / 2.0])
            )  # Arrow starts at centroid and goes into single direction
            self._meshcat.SetTransform(
                path + "/torque/head",
                RigidTransform(RotationMatrix.MakeXRotation(np.pi), [0.0, 0.0, torque_height + arrowhead_height]),
            )

    @staticmethod
    def _get_separation_direction_vec(
        outer_translations: np.ndarray, inner_translations: np.ndarray, viz: bool = False
    ) -> np.ndarray:
        """
        Returns the vector perpendicular to both the manipuland translations and the z-axis where the translations are
        the combined outer and inner translations.

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
            # Visualize the outer translations in green, the inner translations in orange, the principal component in
            # blue, and the separation vector in red
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

    def setup(self) -> None:
        """Sets up the visualizer."""
        self._visualize_manipulands()

        # Add meshcat
        meshcat_params = MeshcatVisualizerParams()
        meshcat_params.role = Role.kProximity
        _ = MeshcatVisualizer.AddToBuilder(
            self._builder, self._scene_graph.get_query_output_port(), self._meshcat, meshcat_params
        )

        # Finalize plant
        self._plant.Finalize()
        diagram = self._builder.Build()

        # Need to simulate for visualization to work
        simulator = Simulator(diagram)
        simulator.AdvanceTo(0.0)

        # Add slider for stepping through time
        self._meshcat.AddSlider(
            name=TIME_SLIDER_NAME,
            min=0.0,
            max=self._times[-1],
            step=self._times[1] - self._times[0],
            value=0.0,
            decrement_keycode="ArrowLeft",
            increment_keycode="ArrowRight",
        )

        # Add buttons for setting item visibility
        self._meshcat.AddButton(name=TOGGLE_INNER_FORCES_BUTTON_NAME)
        self._meshcat.AddButton(name=TOGGLE_OUTER_FORCES_BUTTON_NAME)
        self._meshcat.AddButton(name=TOGGLE_INNER_GENERALIZED_FORCES_BUTTON_NAME)
        self._meshcat.AddButton(name=TOGGLE_OUTER_GENERALIZED_FORCES_BUTTON_NAME)
        self._meshcat.AddButton(name=TOGGLE_INNER_MANIPULAND_BUTTON_NAME)
        self._meshcat.AddButton(name=TOGGLE_OUTER_MANIPULAND_BUTTON_NAME)

        self._is_setup = True

    def _update_item_visibility(self) -> None:
        """Updates item visibility based on button clicks."""
        if self._meshcat.GetButtonClicks(TOGGLE_INNER_FORCES_BUTTON_NAME) > self._toggle_inner_forces_button_clicks:
            self._toggle_inner_forces_button_clicks += 1
            self._inner_forces_visible = not self._inner_forces_visible
        self._meshcat.SetProperty("contact_forces/inner_sim", "visible", self._inner_forces_visible)
        if self._meshcat.GetButtonClicks(TOGGLE_OUTER_FORCES_BUTTON_NAME) > self._toggle_outer_forces_button_clicks:
            self._toggle_outer_forces_button_clicks += 1
            self._outer_forces_visible = not self._outer_forces_visible
        self._meshcat.SetProperty("contact_forces/outer_sim", "visible", self._outer_forces_visible)
        if (
            self._meshcat.GetButtonClicks(TOGGLE_INNER_GENERALIZED_FORCES_BUTTON_NAME)
            > self._toggle_inner_generalized_forces_button_clicks
        ):
            self._toggle_inner_generalized_forces_button_clicks += 1
            self._inner_generalized_forces_visible = not self._inner_generalized_forces_visible
        self._meshcat.SetProperty(
            "contact_forces/inner_sim_generalized", "visible", self._inner_generalized_forces_visible
        )
        if (
            self._meshcat.GetButtonClicks(TOGGLE_OUTER_GENERALIZED_FORCES_BUTTON_NAME)
            > self._toggle_outer_generalized_forces_button_clicks
        ):
            self._toggle_outer_generalized_forces_button_clicks += 1
            self._outer_generalized_forces_visible = not self._outer_generalized_forces_visible
        self._meshcat.SetProperty(
            "contact_forces/outer_sim_generalized", "visible", self._outer_generalized_forces_visible
        )
        if (
            self._meshcat.GetButtonClicks(TOGGLE_INNER_MANIPULAND_BUTTON_NAME)
            > self._toggle_inner_manipuland_button_clicks
        ):
            self._toggle_inner_manipuland_button_clicks += 1
            self._inner_manipuland_visible = not self._inner_manipuland_visible
        self._meshcat.SetProperty(f"visualizer/inner_manipuland", "visible", self._inner_manipuland_visible)
        if (
            self._meshcat.GetButtonClicks(TOGGLE_OUTER_MANIPULAND_BUTTON_NAME)
            > self._toggle_outer_manipuland_button_clicks
        ):
            self._toggle_outer_manipuland_button_clicks += 1
            self._outer_manipuland_visible = not self._outer_manipuland_visible
        self._meshcat.SetProperty(f"visualizer/{self._manipuland_name}", "visible", self._outer_manipuland_visible)

    def _visualize_generalized_contact_forces(self, time_idx: int) -> None:
        outer_generalized_contact_force = self._outer_generalized_contact_forces[time_idx]
        if np.linalg.norm(outer_generalized_contact_force) > self._force_magnitude_theshold:
            self._add_single_direction_force_arrow(
                path="contact_forces/outer_sim_generalized",
                force=outer_generalized_contact_force[3:],
                torque=outer_generalized_contact_force[:3],
                centroid=self._outer_manipuland_poses[time_idx][4:],
                force_rgba=Rgba(0.0, 0.0, 1.0, 1.0),  # blue
                torque_rgba=Rgba(0.6, 0.6, 1.0, 1.0),  # purple
            )
        inner_generalized_contact_force = self._inner_generalized_contact_forces[time_idx]
        if np.linalg.norm(inner_generalized_contact_force) > self._force_magnitude_theshold:
            self._add_single_direction_force_arrow(
                path="contact_forces/inner_sim_generalized",
                force=inner_generalized_contact_force[3:],
                torque=inner_generalized_contact_force[:3],
                centroid=self._inner_manipuland_poses[time_idx][4:],
                force_rgba=Rgba(1.0, 0.0, 0.5, 1.0),  # pink
                torque_rgba=Rgba(1.0, 0.6, 0.8, 1.0),  # light pink
            )

    def _visualize_contact_forces(self, time_idx: int) -> None:
        if self._hydroelastic:
            # NOTE: The hydroelastic forces seem very different to the ones in the recorded HTMLs of the simulation.
            # Further investigation is needed to determine why this is the case. For now, it is better to use this visualizer
            # for point contact visualizations.
            for i, (force, torque, centroid) in enumerate(
                zip(
                    self._outer_hydroelastic_contact_forces[time_idx],
                    self._outer_hydroelastic_contact_torques[time_idx],
                    self._outer_hydroelastic_centroids[time_idx],
                )
            ):
                if np.linalg.norm(force) > self._force_magnitude_theshold:
                    self._add_single_direction_force_arrow(
                        path=f"contact_forces/outer_sim/force_{i}",
                        force=force,
                        torque=torque,
                        centroid=centroid,
                        force_rgba=Rgba(0.0, 1.0, 0.0, 1.0),
                        torque_rgba=Rgba(0.0, 1.0, 1.0, 1.0),  # light blue
                    )

            for i, (force, torque, centroid) in enumerate(
                zip(
                    self._inner_hydroelastic_contact_forces[time_idx],
                    self._inner_hydroelastic_contact_torques[time_idx],
                    self._inner_hydroelastic_centroids[time_idx],
                )
            ):
                if np.linalg.norm(force) > self._force_magnitude_theshold:
                    self._add_single_direction_force_arrow(
                        path=f"contact_forces/inner_sim/force_{i}",
                        force=force,
                        torque=torque,
                        centroid=centroid,
                        force_rgba=Rgba(1.0, 0.0, 0.0, 1.0),
                        torque_rgba=Rgba(1.0, 0.5, 0.0, 1.0),  # orange
                    )
        else:
            for i, (force, point) in enumerate(
                zip(self._outer_point_contact_forces[time_idx], self._outer_point_contact_points[time_idx])
            ):
                if np.linalg.norm(force) > self._force_magnitude_theshold:
                    self._add_equal_opposite_force_arrow(
                        path=f"contact_forces/outer_sim/force_{i}",
                        force=force,
                        contact_point=point,
                        rgba=Rgba(0.0, 1.0, 0.0, 1.0),
                    )

            for i, (force, point) in enumerate(
                zip(self._inner_point_contact_forces[time_idx], self._inner_point_contact_points[time_idx])
            ):
                if np.linalg.norm(force) > self._force_magnitude_theshold:
                    self._add_equal_opposite_force_arrow(
                        path=f"contact_forces/inner_sim/force_{i}",
                        force=force,
                        contact_point=point,
                        rgba=Rgba(1.0, 0.0, 0.0, 1.0),
                    )

    def _update_manipuland_poses(self, time_idx: int) -> None:
        self._meshcat.SetTransform(
            f"visualizer/{self._manipuland_name}/{self._manipuland_base_link_name}",
            vector_pose_to_rigidtransform(self._outer_manipuland_poses[time_idx]),
        )
        self._meshcat.SetTransform(
            f"visualizer/inner_manipuland/{self._manipuland_base_link_name}",
            vector_pose_to_rigidtransform(self._inner_manipuland_poses[time_idx]),
        )

    def _save_current_html(self) -> None:
        """Saves the HTML of the current timestep. NOTE: This overrides the previously saved HTML."""
        html = self._meshcat.StaticHtml()
        html_path = os.path.join(self._data_path, "contact_force_visualizer.html")
        with open(html_path, "w") as f:
            f.write(html)

    def _run_loop_iteration(self, current_time: int) -> Tuple[float, int]:
        """
        :param current_time: The current timestep.
        :return: A tuple of (updated current_time, time_idx for new current_time).
        """
        new_demanded_time = self._meshcat.GetSliderValue(TIME_SLIDER_NAME)
        time_idx = np.abs(self._times - new_demanded_time).argmin()
        new_time = self._times[time_idx]

        # Only update meshcat if the data changed
        if new_time == current_time:
            time.sleep(0.1)
            return current_time, time_idx
        else:
            current_time = new_time

        # Delete all force arrows before adding the new ones
        self._meshcat.Delete("contact_forces")

        self._update_item_visibility()

        self._visualize_generalized_contact_forces(time_idx)

        self._visualize_contact_forces(time_idx)

        self._update_manipuland_poses(time_idx)

        return current_time, time_idx

    def run(self) -> None:
        """Runs the infinite visualizer loop."""
        assert self._is_setup, "`setup()` must be called before calling `run()`."

        print("Entering infinite loop. Force quit to exit.")  # TODO: Check for user input in non-blocking way
        current_time = -1.0  # Force initial meshcat update
        while True:
            current_time, _ = self._run_loop_iteration(current_time)

            if self._save_html:
                self._save_current_html()
