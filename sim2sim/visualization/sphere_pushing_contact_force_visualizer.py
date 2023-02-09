import os

import numpy as np
from pydrake.all import Sphere, RigidTransform
from manipulation.scenarios import AddShape

from sim2sim.visualization import ContactForceVisualizer

TOGGLE_INNER_SPHERE_BUTTON_NAME = "Toggle inner sphere default visibility"
TOGGLE_OUTER_SPHERE_BUTTON_NAME = "Toggle outer sphere default visibility"


class SpherePushingContactForceVisualizer(ContactForceVisualizer):
    """A sphere pushing experiment specific contact force visualizer."""

    def __init__(
        self,
        data_path: str,
        manipuland: str,
        separation_distance: float,
        save_html: bool,
        newtons_per_meter: float,
        newton_meters_per_meter: float,
        hydroelastic: bool,
        sphere_transparency: float,
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
        :param sphere_transparency: The alpha value of the sphere in range (0,1].
        """
        super().__init__(
            data_path,
            manipuland,
            separation_distance,
            save_html,
            newtons_per_meter,
            newton_meters_per_meter,
            hydroelastic,
        )
        self._sphere_transparency = sphere_transparency

        self._sphere_radius = self._experiment_description["script"]["args"]["sphere_radius"]

        # Meshcat button data
        self._toggle_inner_sphere_button_clicks = 0
        self._inner_sphere_visible = True
        self._toggle_outer_sphere_button_clicks = 0
        self._outer_sphere_visible = True

        # Load sphere data
        self._outer_sphere_translations = np.loadtxt(os.path.join(self._log_dir, f"outer_sphere_poses.txt"))[:, :3]
        self._inner_sphere_translations = np.loadtxt(os.path.join(self._log_dir, f"inner_sphere_poses.txt"))[:, :3]

        if self._manipuland == "both" and self._separation_distance > 0.0:
            self._modify_sphere_data_for_side_by_side_visualization()

    def _modify_sphere_data_for_side_by_side_visualization(self) -> None:
        self._outer_sphere_translations += self._separation_vec
        self._inner_sphere_translations -= self._separation_vec

    def _visualize_spheres(self) -> None:
        """Visualizes the sphere(s) at the world origin."""
        # TODO: Debug why `sphere_transparency` doesn't do anything
        if self._manipuland in ["outer", "both"]:
            AddShape(
                self._plant,
                Sphere(self._sphere_radius),
                "outer_sphere",
                color=[0.9, 0.5, 0.5, self._sphere_transparency],
            )
        if self._manipuland in ["inner", "both"]:
            AddShape(
                self._plant,
                Sphere(self._sphere_radius),
                "inner_sphere",
                color=[0.9, 0.5, 0.5, self._sphere_transparency],
            )

    def _update_sphere_poses(self, time_idx: int) -> None:
        self._meshcat.SetTransform(
            f"visualizer/outer_sphere", RigidTransform(p=self._outer_sphere_translations[time_idx])
        )
        self._meshcat.SetTransform(
            f"visualizer/inner_sphere", RigidTransform(p=self._inner_sphere_translations[time_idx])
        )

    def _update_item_visibility(self) -> None:
        if self._meshcat.GetButtonClicks(TOGGLE_INNER_SPHERE_BUTTON_NAME) > self._toggle_inner_sphere_button_clicks:
            self._toggle_inner_sphere_button_clicks += 1
            self._inner_sphere_visible = not self._inner_sphere_visible
        self._meshcat.SetProperty(f"visualizer/inner_sphere", "visible", self._inner_sphere_visible)
        if self._meshcat.GetButtonClicks(TOGGLE_OUTER_SPHERE_BUTTON_NAME) > self._toggle_outer_sphere_button_clicks:
            self._toggle_outer_sphere_button_clicks += 1
            self._outer_sphere_visible = not self._outer_sphere_visible
        self._meshcat.SetProperty(f"visualizer/outer_sphere", "visible", self._outer_sphere_visible)

        super()._update_item_visibility()

    def setup(self) -> None:
        self._visualize_spheres()

        super().setup()

        self._meshcat.AddButton(name=TOGGLE_INNER_SPHERE_BUTTON_NAME)
        self._meshcat.AddButton(name=TOGGLE_OUTER_SPHERE_BUTTON_NAME)

    def _run_loop_iteration(self, current_time: int) -> int:
        new_time, time_idx = super()._run_loop_iteration(current_time)
        if new_time != current_time:
            self._update_sphere_poses(time_idx)
        return new_time, time_idx
