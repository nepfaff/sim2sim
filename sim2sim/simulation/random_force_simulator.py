import os
import time

from typing import Optional

import numpy as np
import open3d as o3d

from pydrake.all import DiagramBuilder, RigidTransform, SceneGraph, Simulator
from scipy.spatial.transform import Rotation as R

from sim2sim.images import generate_camera_locations_sphere
from sim2sim.logging import DynamicLogger
from sim2sim.util import ExternalForceSystem

from .simulator_base import SimulatorBase


class RandomForceSimulator(SimulatorBase):
    """A simulator that uses a point finger to apply a random force to the manipuland."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLogger,
        is_hydroelastic: bool,
        use_point_finger: bool,
        force_magnitude: float,
        settling_time: float,
        manipuland_name: str,
        mesh_path: Optional[str] = None,
        random_seed: int = None,
        skip_outer_visualization: bool = False,
    ):
        """
        :param outer_builder: Diagram builder for the outer simulation environment.
        :param outer_scene_graph: Scene graph for the outer simulation environment.
        :param inner_builder: Diagram builder for the inner simulation environment.
        :param inner_scene_graph: Scene graph for the inner simulation environment.
        :param logger: The logger.
        :param is_hydroelastic: Whether hydroelastic or point contact is used.
        :param use_point_finger: Whether to use a point finger to apply the force. Otherwise, the external force will
            be applied onto the object directly.
        :param force_magnitude: The magnitude of the force to apply in N.
        :param settling_time: The time in seconds to simulate initially to allow the scene to settle.
        :param manipuland_name: The name of the manipuland model instance.
        :param mesh_path: The path to the visual mesh that is used for selecting the point to apply force to. Only
            needed if `use_point_finger` is `False`.
        :param random_seed: The random seed to use. It will be random, if None.
        """
        super().__init__(
            outer_builder,
            outer_scene_graph,
            inner_builder,
            inner_scene_graph,
            logger,
            is_hydroelastic,
            skip_outer_visualization,
        )

        self._use_point_finger = use_point_finger
        self._force_magnitude = force_magnitude
        self._settling_time = settling_time
        self._manipuland_name = manipuland_name
        self._mesh_path = mesh_path

        self._random_generator = (
            np.random.default_rng()
            if random_seed is None
            else np.random.default_rng(random_seed)
        )

        self._finalize_and_build_diagrams()

    def _finalize_and_build_diagrams(self) -> None:
        """Adds visualization systems to the outer and inner diagrams and builds them."""
        if self._skip_outer_visualization:
            self._outer_visualizer, self._outer_meshcat = None, None
        else:
            self._outer_visualizer, self._outer_meshcat = self._logger.add_visualizers(
                self._outer_builder,
                self._outer_scene_graph,
                self._is_hydroelastic,
                is_outer=True,
            )
        self._inner_visualizer, self._inner_meshcat = self._logger.add_visualizers(
            self._inner_builder,
            self._inner_scene_graph,
            self._is_hydroelastic,
            is_outer=False,
        )

        self._logger.add_manipuland_pose_logging(
            self._outer_builder, self._inner_builder
        )
        self._logger.add_manipuland_contact_force_logging(
            self._outer_builder, self._inner_builder
        )
        self._logger.add_contact_result_logging(
            self._outer_builder, self._inner_builder
        )

        self._outer_diagram = self._outer_builder.Build()
        self._inner_diagram = self._inner_builder.Build()

    def simulate(self, duration: float) -> None:
        for i, (diagram, visualizer, meshcat) in enumerate(
            zip(
                [self._outer_diagram, self._inner_diagram],
                [self._outer_visualizer, self._inner_visualizer],
                [self._outer_meshcat, self._inner_meshcat],
            )
        ):
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()

            diagram.get_input_port().FixValue(context, [0.0, 0.0, 0.0])

            if i == 1 or not self._skip_outer_visualization:
                # TODO: Move `StartRecording` and `StopRecording` into logger using
                # `with` statement
                visualizer.StartRecording()

            start_time = time.time()

            simulator.AdvanceTo(self._settling_time)

            plant = diagram.GetSubsystemByName("plant")
            plant_context = plant.GetMyContextFromRoot(context)
            mesh_manipuland_instance = plant.GetModelInstanceByName(
                self._manipuland_name
            )
            mesh_manipuland_pose_vec = plant.GetPositions(
                plant_context, mesh_manipuland_instance
            )
            mesh_manipuland_translation = mesh_manipuland_pose_vec[4:]
            mesh_manipuland_orientation = mesh_manipuland_pose_vec[:4]

            if i == 0:
                if self._use_point_finger:
                    # Pick a random finger start location on a half-sphere around the manipuland
                    finger_start_locations = generate_camera_locations_sphere(
                        center=mesh_manipuland_translation
                        - [0.0, 0.0, mesh_manipuland_pose_vec[-1] / 2.0],
                        radius=0.3,
                        num_phi=10,
                        num_theta=30,
                        half=True,
                    )
                    finger_start_location = finger_start_locations[
                        self._random_generator.choice(len(finger_start_locations))
                    ]
                    direction = finger_start_location - mesh_manipuland_translation
                else:
                    mesh = o3d.io.read_triangle_mesh(self._mesh_path)
                    vertices = np.asarray(mesh.vertices)
                    random_idx = self._random_generator.choice(len(vertices))
                    vertex_wrt_mesh_manipuland = vertices[random_idx]
                    normal_wrt_mesh_manipuland = np.asarray(mesh.vertex_normals)[
                        random_idx
                    ]

                    X_WorldManipuland = RigidTransform()
                    X_WorldManipuland = np.eye(4)
                    X_WorldManipuland[:3, :3] = R.from_quat(
                        mesh_manipuland_orientation
                    ).as_matrix()
                    X_WorldManipuland[:3, 3] = mesh_manipuland_translation

                    vertex_wrt_world = (
                        X_WorldManipuland[:3, :3] @ vertex_wrt_mesh_manipuland
                        + X_WorldManipuland[:3, 3]
                    )
                    normal_wrt_world = (
                        X_WorldManipuland[:3, :3] @ normal_wrt_mesh_manipuland
                        + X_WorldManipuland[:3, 3]
                    )

                    direction = vertex_wrt_world - normal_wrt_world

            if self._use_point_finger:
                plant_context = plant.GetMyContextFromRoot(context)
                point_finger = plant.GetModelInstanceByName("point_finger")
                plant.SetPositions(plant_context, point_finger, finger_start_location)
                force = self._force_magnitude / np.linalg.norm(direction) * direction
                diagram.get_input_port().FixValue(context, force)
            else:
                external_force_system: ExternalForceSystem = diagram.GetSubsystemByName(
                    "external_force_system"
                )
                force = self._force_magnitude / np.linalg.norm(direction) * direction
                torque = np.zeros(3)
                external_force_system.set_wrench(np.concatenate([force, torque]))
                external_force_system.set_wrench_application_point(vertex_wrt_world)

            simulator.AdvanceTo(self._settling_time + duration)

            time_taken_to_simulate = time.time() - start_time
            if i == 0:
                self._logger.log(outer_simulation_time=time_taken_to_simulate)
            else:
                self._logger.log(inner_simulation_time=time_taken_to_simulate)

            if i == 1 or not self._skip_outer_visualization:
                visualizer.StopRecording()
                visualizer.PublishRecording()

                # TODO: Move this to the logger
                html = meshcat.StaticHtml()
                with open(
                    os.path.join(
                        self._logger._logging_path,
                        f"{'inner' if i == 1 else 'outer'}.html",
                    ),
                    "w",
                ) as f:
                    f.write(html)

            context = simulator.get_mutable_context()
            self._logger.log_manipuland_poses(context, is_outer=(i == 0))
            self._logger.log_manipuland_contact_forces(context, is_outer=(i == 0))
