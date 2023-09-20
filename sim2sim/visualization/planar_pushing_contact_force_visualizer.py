import os

import numpy as np

from manipulation.scenarios import AddShape
from pydrake.all import Box, RigidTransform, Sphere

from sim2sim.visualization import ContactForceVisualizer

TOGGLE_INNER_PUSHER_GEOMETRY_BUTTON_NAME = (
    "Toggle inner pusher geometry default visibility"
)
TOGGLE_OUTER_PUSHER_GEOMETRY_BUTTON_NAME = (
    "Toggle outer pusher geometry default visibility"
)


class PlanarPushingContactForceVisualizer(ContactForceVisualizer):
    """A planar pushing experiment specific contact force visualizer."""

    def __init__(
        self,
        data_path: str,
        manipuland: str,
        separation_distance: float,
        save_html: bool,
        newtons_per_meter: float,
        newton_meters_per_meter: float,
        hydroelastic: bool,
        kIllustration: bool,
        pusher_geometry_transparency: float,
    ):
        """
        :param data_path: Path to the experiment data folder.
        :param manipuland: The manipuland to visualize. Options are 'outer', 'inner',
            'both', and 'none'.
        :param separation_distance: The distance in meters that the outer and inner
            manipuland should be separated from each other. This only has an effect if
            `--manipuland` is 'both'.
        :param save_html: Whether to save the meshcat HTML.
        :param newtons_per_meter: Sets the length scale of the force vectors.
        :param newton_meters_per_meter: Sets the length scale of the torque/ moment
            vectors.
        :param hydroelastic: Whether to plot hydroelastic or point contact forces.
        :param kIllustration: Whether to use kIllustration or kProximity for meshcat.
        :param pusher_geometry_transparency: The alpha value of the pusher geometry in
            range (0,1].
        """
        super().__init__(
            data_path,
            manipuland,
            separation_distance,
            save_html,
            newtons_per_meter,
            newton_meters_per_meter,
            hydroelastic,
            kIllustration,
        )
        self._pusher_geometry_transparency = pusher_geometry_transparency

        self._pusher_geometry_type = self._experiment_description["script"]["args"][
            "pusher_geometry_type"
        ]
        self._pusher_geometry_dimenensions = self._experiment_description["script"][
            "args"
        ]["pusher_geometry_dimensions"]

        # Meshcat button data
        self._toggle_inner_pusher_geometry_button_clicks = 0
        self._inner_pusher_geometry_visible = True
        self._toggle_outer_pusher_geometry_button_clicks = 0
        self._outer_pusher_geometry_visible = True

        # Load pusher_geometry data
        self._outer_pusher_geometry_translations = np.loadtxt(
            os.path.join(self._log_dir, f"outer_pusher_geometry_poses.txt")
        )[:, :3]
        self._inner_pusher_geometry_translations = np.loadtxt(
            os.path.join(self._log_dir, f"inner_pusher_geometry_poses.txt")
        )[:, :3]

        if self._manipuland == "both" and self._separation_distance > 0.0:
            self._modify_pusher_geometry_data_for_side_by_side_visualization()

    def _modify_pusher_geometry_data_for_side_by_side_visualization(self) -> None:
        self._outer_pusher_geometry_translations += self._separation_vec
        self._inner_pusher_geometry_translations -= self._separation_vec

    def _visualize_pusher_geometries(self) -> None:
        """Visualizes the pusher geometry/geometries at the world origin."""
        # TODO: Debug why `pusher_geometry_transparency` doesn't do anything

        pusher_geometry = (
            Sphere(self._pusher_geometry_dimenensions)
            if self._pusher_geometry_type.lower() == "sphere"
            else Box(
                self._pusher_geometry_dimenensions[0],
                self._pusher_geometry_dimenensions[1],
                self._pusher_geometry_dimenensions[2],
            )
        )

        add_shape = lambda name: AddShape(
            self._plant,
            pusher_geometry,
            name,
            color=[0.9, 0.5, 0.5, self._pusher_geometry_transparency],
        )
        if self._manipuland in ["outer", "both"]:
            add_shape("outer_pusher_geometry")
        if self._manipuland in ["inner", "both"]:
            add_shape("inner_pusher_geometry")

    def _update_pusher_geometry_poses(self, time_idx: int) -> None:
        self._meshcat.SetTransform(
            f"visualizer/outer_pusher_geometry",
            RigidTransform(p=self._outer_pusher_geometry_translations[time_idx]),
        )
        self._meshcat.SetTransform(
            f"visualizer/inner_pusher_geometry",
            RigidTransform(p=self._inner_pusher_geometry_translations[time_idx]),
        )

    def _update_item_visibility(self) -> None:
        if (
            self._meshcat.GetButtonClicks(TOGGLE_INNER_PUSHER_GEOMETRY_BUTTON_NAME)
            > self._toggle_inner_pusher_geometry_button_clicks
        ):
            self._toggle_inner_pusher_geometry_button_clicks += 1
            self._inner_pusher_geometry_visible = (
                not self._inner_pusher_geometry_visible
            )
        self._meshcat.SetProperty(
            f"visualizer/inner_pusher_geometry",
            "visible",
            self._inner_pusher_geometry_visible,
        )
        if (
            self._meshcat.GetButtonClicks(TOGGLE_OUTER_PUSHER_GEOMETRY_BUTTON_NAME)
            > self._toggle_outer_pusher_geometry_button_clicks
        ):
            self._toggle_outer_pusher_geometry_button_clicks += 1
            self._outer_pusher_geometry_visible = (
                not self._outer_pusher_geometry_visible
            )
        self._meshcat.SetProperty(
            f"visualizer/outer_pusher_geometry",
            "visible",
            self._outer_pusher_geometry_visible,
        )

        super()._update_item_visibility()

    def setup(self) -> None:
        self._visualize_pusher_geometries()

        super().setup()

        self._meshcat.AddButton(name=TOGGLE_INNER_PUSHER_GEOMETRY_BUTTON_NAME)
        self._meshcat.AddButton(name=TOGGLE_OUTER_PUSHER_GEOMETRY_BUTTON_NAME)

    def _run_loop_iteration(self, current_time: int) -> int:
        new_time, time_idx = super()._run_loop_iteration(current_time)
        if new_time != current_time:
            self._update_pusher_geometry_poses(time_idx)
        return new_time, time_idx
