#!/bin/python3

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# TODO: Move into main codebase
class ContactForceVisualizer:
    def __init__(self):
        self._centroids = np.load(
            "/home/nep/robot_locomotion/sim2sim/logs/iiwa_manip_rearrangement_pepper/time_logs/outer_contact_result_centroids.npy",
            allow_pickle=True,
        )
        self._forces = np.load(
            "/home/nep/robot_locomotion/sim2sim/logs/iiwa_manip_rearrangement_pepper/time_logs/outer_contact_result_forces.npy",
            allow_pickle=True,
        )
        self._times = np.load(
            "/home/nep/robot_locomotion/sim2sim/logs/iiwa_manip_rearrangement_pepper/time_logs/outer_contact_result_times.npy"
        )

        self._logging_frequency = 100

        self._max_time = self._times[-1]
        self._create_layout(int(self._max_time * self._logging_frequency))

    def _create_layout(self, max_time: float) -> None:
        self.window = gui.Application.instance.create_window("Contact Force Visualizer", 1024, 768)

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)

        em = self.window.theme.font_size
        self._panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self._slider = gui.Slider(gui.Slider.INT)
        self._slider.set_limits(0.0, max_time)
        self._slider.set_on_value_changed(self._on_slider)
        self._panel.add_child(self._slider)

        doubleedit = gui.NumberEdit(gui.NumberEdit.INT)
        doubleedit.set_on_value_changed(self._on_edit)
        self._panel.add_child(doubleedit)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._panel)

    def _on_slider(self, new_val):
        self._scene.scene.clear_geometry()

        material = o3d.visualization.rendering.MaterialRecord()

        idx = int(new_val)
        centroids, forces = self._centroids[idx], self._forces[idx]
        for i, (centroid, force) in enumerate(zip(centroids, forces)):
            print(i, centroid)
            force_arrow = get_arrow(centroid, vec=force)
            force_arrow.paint_uniform_color(np.array([1.0, 0.0, 0.0]))
            self._scene.scene.add_geometry(f"force_arrow{i}", force_arrow, material)

    def _on_edit(self, new_val):
        if new_val < 0:
            new_val = 0
        if new_val > self._max_time:
            new_val = self._max_time
        self._slider.double_value = new_val

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(r.height, self._panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self._panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)


# TODO: Clean these up
def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec ():
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return magnitude


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the
    z axis vector of the original FOR. The first rotation that is
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis.

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec ():
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1] / vec[0])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T @ vec.reshape(-1, 1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0] / vec[2])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    return (Rz, Ry)


def create_arrow(scale=0.1):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale * 0.2
    cylinder_height = scale * 0.8
    cone_radius = scale / 10
    cylinder_radius = scale / 20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
    )
    return mesh_frame


def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return mesh


def main():
    gui.Application.instance.initialize()
    ContactForceVisualizer()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
