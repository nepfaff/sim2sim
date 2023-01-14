import os
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pydrake.all import (
    DiagramBuilder,
    SceneGraph,
    Simulator,
    RigidTransform,
)

from sim2sim.logging import DynamicLoggerBase
from sim2sim.simulation import SimulatorBase
from sim2sim.images import generate_camera_locations_sphere


class RandomForceSimulator(SimulatorBase):
    """A simulator that uses a point finger to apply a random force to the manipuland."""

    def __init__(
        self,
        outer_builder: DiagramBuilder,
        outer_scene_graph: SceneGraph,
        inner_builder: DiagramBuilder,
        inner_scene_graph: SceneGraph,
        logger: DynamicLoggerBase,
        mesh_path: str,
        inner_only: bool,
    ):
        """
        :param outer_builder: Diagram builder for the outer simulation environment.
        :param outer_scene_graph: Scene graph for the outer simulation environment.
        :param inner_builder: Diagram builder for the inner simulation environment.
        :param inner_scene_graph: Scene graph for the inner simulation environment.
        :param logger: The logger.
        :param mesh_path: The path to the visual mesh that is used for selecting the point to apply force to.
        :param inner_only: Whether to only simulate the inner environment.
        """
        super().__init__(outer_builder, outer_scene_graph, inner_builder, inner_scene_graph, logger)

        self._mesh_path = mesh_path
        self._inner_only = inner_only

        self._finalize_and_build_diagrams()

    def _finalize_and_build_diagrams(self) -> None:
        """Adds visualization systems to the outer and inner diagrams and builds them."""
        if not self._inner_only:
            self._outer_visualizer, self._outer_meshcat = self._logger.add_visualizers(
                self._outer_builder,
                self._outer_scene_graph,
                is_outer=True,
            )
        self._inner_visualizer, self._inner_meshcat = self._logger.add_visualizers(
            self._inner_builder,
            self._inner_scene_graph,
            is_outer=False,
        )

        self._logger.add_manipuland_pose_logging(self._outer_builder, self._inner_builder)
        self._logger.add_manipuland_contact_force_logging(self._outer_builder, self._inner_builder)
        self._logger.add_contact_result_logging(self._outer_builder, self._inner_builder)

        self._outer_diagram = self._outer_builder.Build()
        self._inner_diagram = self._inner_builder.Build()

    def simulate(self, duration: float) -> None:
        mesh = o3d.io.read_triangle_mesh(self._mesh_path)
        vertices = np.asarray(mesh.vertices)
        vertex_wrt_mesh_manipuland = vertices[np.random.choice(len(vertices))]
        normal_wrt_mesh_manipuland = np.asarray(mesh.vertex_normals)[np.random.choice(len(vertices))]

        for i, (diagram, visualizer, meshcat) in enumerate(
            zip(
                [self._inner_diagram, *([] if self._inner_only else [self._outer_diagram])],
                [self._inner_visualizer, *([] if self._inner_only else [self._outer_visualizer])],
                [self._inner_meshcat, *([] if self._inner_only else [self._outer_meshcat])],
            )
        ):
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()

            diagram.get_input_port().FixValue(context, [0.0, 0.0, 0.0])

            # 1. Pick random mesh vertex
            # Need to get triangular mesh + its pose from scene graph. Can then pick vertex + find its pose
            # 2. Find vertex normal
            # 3. Apply force using point finger (see manipulation 'point_finger.ipynb' example)
            #   -start finger at normal and apply force in direction of vertex
            #   -other option would be to sample points along half sphere for finger starting position + sample point within
            #   mesh volume instead of vertex or use nearest mesh vertex

            # Simulate for scene to settle (TODO: Make a param)
            settling_time = 0.3
            simulator.AdvanceTo(settling_time)

            plant = diagram.GetSubsystemByName("plant")
            plant_context = plant.GetMyContextFromRoot(context)
            mesh_manipuland_instance = plant.GetModelInstanceByName("ycb_tomato_soup_can")  # TODO: make argument
            mesh_manipuland_pose_vec = plant.GetPositions(plant_context, mesh_manipuland_instance)
            mesh_manipuland_pose = RigidTransform()
            mesh_manipuland_pose = np.eye(4)
            mesh_manipuland_pose[:3, :3] = R.from_quat(mesh_manipuland_pose_vec[:4]).as_matrix()
            mesh_manipuland_pose[:3, 3] = mesh_manipuland_pose_vec[4:]
            # vertex_wrt_world = mesh_manipuland_pose[:3, :3] @ vertex_wrt_mesh_manipuland + mesh_manipuland_pose[:3, 3]
            # normal_wrt_world = mesh_manipuland_pose[:3, :3] @ normal_wrt_mesh_manipuland + mesh_manipuland_pose[:3, 3]

            print("pos2", mesh_manipuland_pose_vec[4:])

            # Pick a random finger start location on a sphere around the manipuland
            finger_start_locations = generate_camera_locations_sphere(
                center=mesh_manipuland_pose_vec[4:] - [0.0, 0.0, mesh_manipuland_pose_vec[-1] / 2.0],
                radius=0.3,
                num_phi=10,
                num_theta=30,
                half=True,
            )
            finger_start_location = finger_start_locations[np.random.choice(len(finger_start_locations))]
            plant_context = plant.GetMyContextFromRoot(context)
            point_finger = plant.GetModelInstanceByName("point_finger")
            plant.SetPositions(plant_context, point_finger, finger_start_location)

            # viz_geoms = [mesh.transform(mesh_manipuland_pose)]
            # for location in finger_start_locations:
            #     viz_geoms.append(o3d.geometry.TriangleMesh.create_sphere(0.3 / 40.0).translate(location))
            # o3d.visualization.draw_geometries(viz_geoms)
            from pydrake.all import Sphere, RotationMatrix

            for i, location in enumerate(finger_start_locations):
                meshcat.SetObject(f"/{i}", Sphere(0.01))
                meshcat.SetTransform(f"/{i}", RigidTransform(RotationMatrix(), location))

            # test
            # vertex_sphere = o3d.geometry.TriangleMesh.create_sphere(0.025).translate(vertex_wrt_world)
            # normal_sphere = o3d.geometry.TriangleMesh.create_sphere(0.025).translate(normal_wrt_world)
            # o3d.visualization.draw_geometries([mesh.transform(mesh_manipuland_pose), vertex_sphere, normal_sphere])
            # end test

            # TODO: Move `StartRecording` and `StopRecording` into logger using `with` statement
            visualizer.StartRecording()

            start_time = time.time()

            # direction = normal_wrt_world - vertex_wrt_world
            direction = finger_start_location - mesh_manipuland_pose_vec[4:]
            print("direction", direction)
            diagram.get_input_port().FixValue(context, 10 / np.linalg.norm(direction) * direction)

            simulator.AdvanceTo(settling_time + duration)

            time_taken_to_simulate = time.time() - start_time
            if i == 0:
                self._logger.log(inner_simulation_time=time_taken_to_simulate)
            else:
                self._logger.log(outer_simulation_time=time_taken_to_simulate)

            visualizer.StopRecording()
            visualizer.PublishRecording()

            # TODO: Move this to the logger
            html = meshcat.StaticHtml()
            with open(os.path.join(self._logger._logging_path, f"{'outer' if i else 'inner'}.html"), "w") as f:
                f.write(html)

            context = simulator.get_mutable_context()
            self._logger.log_manipuland_poses(context, is_outer=(i == 1))
            self._logger.log_manipuland_contact_forces(context, is_outer=(i == 1))
