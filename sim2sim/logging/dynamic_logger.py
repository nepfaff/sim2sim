from typing import Tuple, List, Optional, Dict, Any
import os
import datetime
import yaml

import numpy as np
import open3d as o3d
from pydrake.all import (
    DiagramBuilder,
    SceneGraph,
    Meshcat,
    StartMeshcat,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    ContactVisualizer,
    MultibodyPlant,
    LogVectorOutput,
    Context,
    ContactResults,
)
from PIL import Image
from matplotlib import pyplot as plt

from sim2sim.logging import DynamicLoggerBase
from sim2sim.util import get_hydroelastic_contact_viz_params, get_point_contact_contact_viz_params
from .abstract_value_logger import AbstractValueLogger


class DynamicLogger(DynamicLoggerBase):
    def __init__(
        self,
        logging_frequency_hz: float,
        logging_path: str,
        kProximity: bool,
        label_to_mask: int,
        manipuland_name: str,
        manipuland_base_link_name: str,
    ):
        """
        :param logging_frequency_hz: The frequency at which we want to log at.
        :param logging_path: The path to the directory that we want to write the log files to.
        :param kProximity: Whether to visualize kProximity or kIllustration. Visualize kProximity if true.
        :param label_to_mask: The label that we want to save binary masks for.
        :param manipuland_name: The name of the manipuland. Required for pose logging.
        :param manipuland_base_link_name: The manipuland base link name. Required for contact result force logging.
        """
        super().__init__(logging_frequency_hz, logging_path, kProximity)

        self._label_to_mask = label_to_mask
        self._manipuland_name = manipuland_name
        self._manipuland_base_link_name = manipuland_base_link_name

        # Manipuland pose logs
        self._outer_manipuland_pose_logger = None
        self._outer_manipuland_poses: np.ndarray = None
        self._outer_manipuland_pose_times: np.ndarray = None
        self._inner_manipuland_pose_logger = None
        self._inner_manipuland_poses: np.ndarray = None
        self._inner_manipuland_pose_times: np.ndarray = None

        # Manipuland contact force logs
        self._outer_manipuland_contact_force_logger = None
        self._outer_manipuland_contact_forces: np.ndarray = None
        self._outer_manipuland_contact_force_times: np.ndarray = None
        self._inner_manipuland_contact_force_logger = None
        self._inner_manipuland_contact_forces: np.ndarray = None
        self._inner_manipuland_contact_force_times: np.ndarray = None

        # Contact result logs
        self._outer_contact_result_logger = None
        self._inner_contact_result_logger = None

        # Manipuland physics
        self._manipuland_mass_estimated: float = None
        self._manipuland_inertia_estimated: List[float] = None

    @staticmethod
    def add_meshcat_visualizer(
        builder: DiagramBuilder, scene_graph: SceneGraph, kProximity: bool
    ) -> Tuple[MeshcatVisualizer, Meshcat]:
        """
        Adds a meshcat visualizer to `builder`.

        :param builder: The diagram builder to add the visualizer to.
        :param scene_graph: The scene graph of the scene to visualize.
        :param kProximity: Whether to visualize kProximity or kIllustration. Visualize kProximity if true.
        :return: A tuple of (visualizer, meshcat).
        """
        meshcat = StartMeshcat()
        meshcat_params = MeshcatVisualizerParams()
        meshcat_params.role = Role.kProximity if kProximity else Role.kIllustration
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph.get_query_output_port(), meshcat, meshcat_params
        )
        return visualizer, meshcat

    @staticmethod
    def _add_contact_visualizer(
        builder: DiagramBuilder, meshcat: Meshcat, plant: MultibodyPlant, is_hydroelastic: bool
    ) -> None:
        """
        Adds a contact visualizer to `builder`.

        :param builder: The diagram builder to add the visualizer to.
        :param meshcat: The meshcat that we want to add the contact visualizer to.
        :param plant: The plant for which we want to visualize contact forces.
        :param is_hydroelastic: Whether to use the hydroelastic or point contact configs.
        """
        cparams = get_hydroelastic_contact_viz_params() if is_hydroelastic else get_point_contact_contact_viz_params()
        _ = ContactVisualizer.AddToBuilder(builder, plant, meshcat, cparams)

    def add_visualizers(
        self, builder: DiagramBuilder, scene_graph: SceneGraph, is_hydroelastic: bool, is_outer: bool
    ) -> Tuple[MeshcatVisualizer, Meshcat]:
        """
        Add visualizers.
        :param builder: The diagram builder to add the visualizer to.
        :param scene_graph: The scene graph of the scene to visualize.
        :param is_hydroelastic: Whether to use the hydroelastic or point contact visualizer configs.
        :param is_outer: Whether it is the outer or inner simulation.
        :return: A tuple of (visualizer, meshcat).
        """
        visualizer, meshcat = self.add_meshcat_visualizer(builder, scene_graph, self._kProximity)
        if self._inner_plant is not None and self._outer_plant is not None:
            self._add_contact_visualizer(
                builder, meshcat, self._outer_plant if is_outer else self._inner_plant, is_hydroelastic
            )
        return visualizer, meshcat

    def add_manipuland_pose_logging(self, outer_builder: DiagramBuilder, inner_builder: DiagramBuilder) -> None:
        self._outer_manipuland_pose_logger = LogVectorOutput(
            self._outer_plant.get_state_output_port(self._outer_plant.GetModelInstanceByName(self._manipuland_name)),
            outer_builder,
            1.0 / self._logging_frequency_hz,
        )
        self._inner_manipuland_pose_logger = LogVectorOutput(
            self._inner_plant.get_state_output_port(self._inner_plant.GetModelInstanceByName(self._manipuland_name)),
            inner_builder,
            1.0 / self._logging_frequency_hz,
        )

    def add_contact_result_logging(self, outer_builder: DiagramBuilder, inner_builder: DiagramBuilder) -> None:
        self._outer_contact_result_logger = outer_builder.AddSystem(
            AbstractValueLogger(ContactResults(), self._logging_frequency_hz)
        )
        outer_builder.Connect(
            self._outer_plant.get_contact_results_output_port(),
            self._outer_contact_result_logger.get_input_port(),
        )
        self._inner_contact_result_logger = inner_builder.AddSystem(
            AbstractValueLogger(ContactResults(), self._logging_frequency_hz)
        )
        inner_builder.Connect(
            self._inner_plant.get_contact_results_output_port(),
            self._inner_contact_result_logger.get_input_port(),
        )

    def log_manipuland_poses(self, context: Context, is_outer: bool) -> None:
        # NOTE: This really logs state which is both pose (7,) and spatial velocity (6,)
        assert self._outer_manipuland_pose_logger is not None and self._inner_manipuland_pose_logger is not None

        if is_outer:
            log = self._outer_manipuland_pose_logger.FindLog(context)
            self._outer_manipuland_pose_times = log.sample_times()
            self._outer_manipuland_poses = log.data().T  # Shape (t, 13)
        else:
            log = self._inner_manipuland_pose_logger.FindLog(context)
            self._inner_manipuland_pose_times = log.sample_times()
            self._inner_manipuland_poses = log.data().T  # Shape (t, 13)

    def add_manipuland_contact_force_logging(
        self, outer_builder: DiagramBuilder, inner_builder: DiagramBuilder
    ) -> None:
        self._outer_manipuland_contact_force_logger = LogVectorOutput(
            self._outer_plant.get_generalized_contact_forces_output_port(
                self._outer_plant.GetModelInstanceByName(self._manipuland_name)
            ),
            outer_builder,
            1 / self._logging_frequency_hz,
        )
        self._inner_manipuland_contact_force_logger = LogVectorOutput(
            self._inner_plant.get_generalized_contact_forces_output_port(
                self._inner_plant.GetModelInstanceByName(self._manipuland_name)
            ),
            inner_builder,
            1 / self._logging_frequency_hz,
        )

    def log_manipuland_contact_forces(self, context: Context, is_outer: bool) -> None:
        assert (
            self._outer_manipuland_contact_force_logger is not None
            and self._inner_manipuland_contact_force_logger is not None
        )

        if is_outer:
            log = self._outer_manipuland_contact_force_logger.FindLog(context)
            self._outer_manipuland_contact_force_times = log.sample_times()
            self._outer_manipuland_contact_forces = log.data().T  # Shape (t, 6)
        else:
            log = self._inner_manipuland_contact_force_logger.FindLog(context)
            self._inner_manipuland_contact_force_times = log.sample_times()
            self._inner_manipuland_contact_forces = log.data().T  # Shape (t, 6)

    def log_manipuland_estimated_physics(
        self, manipuland_mass_estimated: float, manipuland_inertia_estimated: np.ndarray
    ) -> None:
        self._manipuland_mass_estimated = float(manipuland_mass_estimated)
        self._manipuland_inertia_estimated = manipuland_inertia_estimated.tolist()

    def _get_contact_result_forces(
        self, is_outer: bool, body_of_interest: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """
        :returns: A tuple of
            - point_contact_contact_result_contact_points
            - point_contact_contact_result_forces
            - hydroelastic_contact_result_centroids
            - hydroelastic_contact_result_forces
            - hydroelastic_contact_result_torques
            - times
        """
        logs: Tuple[List[ContactResults], List[float]] = (
            self._outer_contact_result_logger.get_logs() if is_outer else self._inner_contact_result_logger.get_logs()
        )
        contact_results, times = logs
        plant = self._outer_plant if is_outer else self._inner_plant

        inspector = self._outer_scene_graph.model_inspector() if is_outer else self._inner_scene_graph.model_inspector()

        point_contact_contact_result_contact_points = []
        point_contact_contact_result_forces = []
        hydroelastic_contact_result_centroids = []
        hydroelastic_contact_result_forces = []
        hydroelastic_contact_result_torques = []
        for contact_result in contact_results:
            point_contact_contact_result_contact_point = []
            point_contact_contact_result_force = []
            for i in range(contact_result.num_point_pair_contacts()):
                contact_info_i = contact_result.point_pair_contact_info(i)
                body_ia_index = contact_info_i.bodyA_index()
                body_ib_index = contact_info_i.bodyB_index()
                if (
                    body_of_interest == plant.get_body(body_ib_index).name()
                    or body_of_interest == plant.get_body(body_ia_index).name()
                ):
                    contact_point = contact_info_i.contact_point()
                    point_contact_contact_result_contact_point.append(contact_point)

                    contact_force = contact_info_i.contact_force()
                    point_contact_contact_result_force.append(contact_force)

            point_contact_contact_result_contact_points.append(point_contact_contact_result_contact_point)
            point_contact_contact_result_forces.append(point_contact_contact_result_force)

            hydroelastic_contact_result_centroid = []
            hydroelastic_contact_result_force = []
            hydroelastic_contact_result_torque = []
            for i in range(contact_result.num_hydroelastic_contacts()):
                contact_info_i = contact_result.hydroelastic_contact_info(i)
                contact_surface = contact_info_i.contact_surface()

                body_ia_geometry_id = contact_surface.id_M()
                body_ia_frame_id = inspector.GetFrameId(body_ia_geometry_id)
                body_ia = plant.GetBodyFromFrameId(body_ia_frame_id)
                body_ib_geometry_id = contact_surface.id_N()
                body_ib_frame_id = inspector.GetFrameId(body_ib_geometry_id)
                body_ib = plant.GetBodyFromFrameId(body_ib_frame_id)
                if body_of_interest == body_ia.name() or body_of_interest == body_ib.name():
                    contact_point = contact_surface.centroid()
                    hydroelastic_contact_result_centroid.append(contact_point)

                    contact_spatial_force = contact_info_i.F_Ac_W()
                    contact_force = contact_spatial_force.translational()
                    hydroelastic_contact_result_force.append(contact_force)
                    contact_torque = contact_spatial_force.rotational()
                    hydroelastic_contact_result_torque.append(contact_torque)

            hydroelastic_contact_result_centroids.append(hydroelastic_contact_result_centroid)
            hydroelastic_contact_result_forces.append(hydroelastic_contact_result_force)
            hydroelastic_contact_result_torques.append(hydroelastic_contact_result_torque)

        return (
            point_contact_contact_result_contact_points,
            point_contact_contact_result_forces,
            hydroelastic_contact_result_centroids,
            hydroelastic_contact_result_forces,
            hydroelastic_contact_result_torques,
            times,
        )

    def log(
        self,
        camera_poses: Optional[List[np.ndarray]] = None,
        intrinsics: Optional[List[np.ndarray]] = None,
        images: Optional[List[np.ndarray]] = None,
        depths: Optional[List[np.ndarray]] = None,
        labels: Optional[List[np.ndarray]] = None,
        masks: Optional[List[np.ndarray]] = None,
        raw_mesh: Optional[o3d.geometry.TriangleMesh] = None,
        processed_mesh: Optional[o3d.geometry.TriangleMesh] = None,
        processed_mesh_piece: Optional[List[o3d.geometry.TriangleMesh]] = None,
        outer_simulation_time: Optional[float] = None,
        inner_simulation_time: Optional[float] = None,
        experiment_description: Optional[dict] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().log(
            camera_poses=camera_poses,
            intrinsics=intrinsics,
            images=images,
            depths=depths,
            labels=labels,
            masks=masks,
            raw_mesh=raw_mesh,
            processed_mesh=processed_mesh,
            processed_mesh_piece=processed_mesh_piece,
            outer_simulation_time=outer_simulation_time,
            inner_simulation_time=inner_simulation_time,
            experiment_description=experiment_description,
            meta_data=meta_data,
        )

    def _create_time_series_plots(self) -> None:
        # Create pose error plots
        if self._outer_manipuland_poses is not None and self._inner_manipuland_poses is not None:
            state_error = self._outer_manipuland_poses - self._inner_manipuland_poses
            times = self._outer_manipuland_pose_times
            orientation_error = state_error[:, :4]  # Quaternions
            translation_error = state_error[:, 4:7]
            angular_velocity_error = state_error[:, 7:10]
            translational_velocity_error = state_error[:, 10:]

            plt.plot(times, np.linalg.norm(translation_error, axis=1))
            plt.xlabel("Time (s)")
            plt.ylabel("Translation error magnitude (m)")
            plt.savefig(os.path.join(self._time_logs_dir_path, "translation_error_magnitude.png"))
            plt.close()

            plt.plot(times, np.linalg.norm(orientation_error, axis=1))
            plt.xlabel("Time (s)")
            plt.ylabel("Orientation error magnitude (quaternions)")
            plt.savefig(os.path.join(self._time_logs_dir_path, "orientation_error_magnitude.png"))
            plt.close()

            plt.plot(times, np.linalg.norm(translational_velocity_error, axis=1))
            plt.xlabel("Time (s)")
            plt.ylabel("Translational velocity error magnitude (m/s)")
            plt.savefig(os.path.join(self._time_logs_dir_path, "translational_velocity_error_magnitude.png"))
            plt.close()

            plt.plot(times, np.linalg.norm(angular_velocity_error, axis=1))
            plt.xlabel("Time (s)")
            plt.ylabel("Angular velocity error magnitude (rad/s)")
            plt.savefig(os.path.join(self._time_logs_dir_path, "angular_velocity_error_magnitude.png"))
            plt.close()

        # Create contact force error plots
        if self._outer_manipuland_contact_forces is not None and self._inner_manipuland_contact_forces is not None:
            contact_force_error = self._outer_manipuland_contact_forces - self._inner_manipuland_contact_forces
            contact_force_error_angular = contact_force_error[:, :3]
            contact_force_error_translational = contact_force_error[:, 3:]

            plt.plot(times, np.linalg.norm(contact_force_error, axis=1))
            plt.xlabel("Time (s)")
            plt.ylabel("Generalized contact force error magnitude (N)")
            plt.savefig(os.path.join(self._time_logs_dir_path, "generalized_contact_force_error_magnitude.png"))
            plt.close()

            plt.plot(times, np.linalg.norm(contact_force_error_angular, axis=1))
            plt.xlabel("Time (s)")
            plt.ylabel("Gemeralized angular contact force error magnitude (N)")
            plt.savefig(os.path.join(self._time_logs_dir_path, "generalized_angular_contact_force_error_magnitude.png"))
            plt.close()

            plt.plot(times, np.linalg.norm(contact_force_error_translational, axis=1))
            plt.xlabel("Time (s)")
            plt.ylabel("Generalized translational contact force error magnitude (N)")
            plt.savefig(
                os.path.join(self._time_logs_dir_path, "generalized_translational_contact_force_error_magnitude.png")
            )
            plt.close()

        # Create contact result force error plots
        (
            _,
            outer_point_contact_result_forces,
            _,
            outer_hydroelastic_contact_result_forces,
            _,
            _,
        ) = self._get_contact_result_forces(True, self._manipuland_base_link_name)
        (
            _,
            inner_point_contact_result_forces,
            _,
            inner_hydroelastic_contact_result_forces,
            _,
            _,
        ) = self._get_contact_result_forces(False, self._manipuland_base_link_name)

        def create_spatial_force_error_plot(outer_forces, inner_forces, ylabel, file_name):
            if len(outer_forces) > 0 and len(inner_forces) > 0:
                outer_forces = np.array([forces[0] if forces else [0.0, 0.0, 0.0] for forces in outer_forces])
                inner_forces = np.array([forces[0] if forces else [0.0, 0.0, 0.0] for forces in inner_forces])
                force_error = outer_forces - inner_forces

            plt.plot(times, np.linalg.norm(force_error, axis=1))
            plt.xlabel("Time (s)")
            plt.ylabel(ylabel)
            plt.savefig(os.path.join(self._time_logs_dir_path, file_name))
            plt.close()

        create_spatial_force_error_plot(
            outer_point_contact_result_forces,
            inner_point_contact_result_forces,
            "Point contact force error magnitude (N)",
            "point_contact_force_error_magnitude.png",
        )
        create_spatial_force_error_plot(
            outer_hydroelastic_contact_result_forces,
            inner_hydroelastic_contact_result_forces,
            "Hydroelastic spatial contact force error magnitude (N)",
            "hydroelastic_spatial_contact_force_error_magnitude.png",
        )

    def postprocess_data(self) -> None:
        self._create_time_series_plots()

    def save_mesh_data(self) -> Tuple[str, str]:
        """
        Saves the raw and processed meshes if they exist.

        :return: A tuple of (raw_mesh_file_path, processed_mesh_file_path).
        """
        raw_mesh_file_path = os.path.join(self._mesh_dir_path, "raw_mesh.obj")
        processed_mesh_file_path = os.path.join(self._mesh_dir_path, "processed_mesh.obj")
        if self._raw_mesh:
            o3d.io.write_triangle_mesh(raw_mesh_file_path, self._raw_mesh)
        if self._processed_mesh:
            o3d.io.write_triangle_mesh(processed_mesh_file_path, self._processed_mesh)
        if self._processed_meshes:
            processed_mesh_file_path = os.path.join(self._mesh_dir_path, "processed_mesh")
            for idx, mesh in enumerate(self._processed_meshes):
                o3d.io.write_triangle_mesh(processed_mesh_file_path + f"_piece_{idx}.obj", mesh)
        return raw_mesh_file_path, processed_mesh_file_path

    def save_manipuland_pose_logs(self) -> None:
        if self._outer_manipuland_poses is not None:
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "outer_manipuland_poses.txt"), self._outer_manipuland_poses
            )
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "outer_manipuland_pose_times.txt"),
                self._outer_manipuland_pose_times,
            )
        if self._inner_manipuland_poses is not None:
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "inner_manipuland_poses.txt"), self._inner_manipuland_poses
            )
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "inner_manipuland_pose_times.txt"),
                self._inner_manipuland_pose_times,
            )

    def save_manipuland_contact_force_logs(self) -> None:
        if self._outer_manipuland_contact_forces is not None:
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "outer_manipuland_contact_forces.txt"),
                self._outer_manipuland_contact_forces,
            )
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "outer_manipuland_contact_force_times.txt"),
                self._outer_manipuland_contact_force_times,
            )
        if self._inner_manipuland_contact_forces is not None:
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "inner_manipuland_contact_forces.txt"),
                self._inner_manipuland_contact_forces,
            )
            np.savetxt(
                os.path.join(self._time_logs_dir_path, "inner_manipuland_contact_force_times.txt"),
                self._inner_manipuland_contact_force_times,
            )

    def save_contact_result_force_logs(self, body_name: str) -> None:
        (
            outer_point_contact_contact_result_contact_points,
            outer_point_contact_contact_result_forces,
            outer_hydroelastic_contact_result_centroids,
            outer_hydroelastic_contact_result_forces,
            outer_hydroelastic_contact_result_torques,
            outer_contact_result_times,
        ) = self._get_contact_result_forces(True, body_name)
        if len(outer_contact_result_times) > 0:
            np.save(
                os.path.join(self._time_logs_dir_path, "outer_point_contact_result_contact_points.npy"),
                np.array(outer_point_contact_contact_result_contact_points, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "outer_point_contact_result_forces.npy"),
                np.array(outer_point_contact_contact_result_forces, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "outer_hydroelastic_contact_result_centroids.npy"),
                np.array(outer_hydroelastic_contact_result_centroids, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "outer_hydroelastic_contact_result_forces.npy"),
                np.array(outer_hydroelastic_contact_result_forces, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "outer_hydroelastic_contact_result_torques.npy"),
                np.array(outer_hydroelastic_contact_result_torques, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "outer_contact_result_times.npy"),
                np.array(outer_contact_result_times),
            )
        (
            inner_point_contact_contact_result_contact_points,
            inner_point_contact_contact_result_forces,
            inner_hydroelastic_contact_result_centroids,
            inner_hydroelastic_contact_result_forces,
            inner_hydroelastic_contact_result_torques,
            inner_contact_result_times,
        ) = self._get_contact_result_forces(False, body_name)
        if len(inner_contact_result_times) > 0:
            np.save(
                os.path.join(self._time_logs_dir_path, "inner_point_contact_result_contact_points.npy"),
                np.array(inner_point_contact_contact_result_contact_points, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "inner_point_contact_result_forces.npy"),
                np.array(inner_point_contact_contact_result_forces, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "inner_hydroelastic_contact_result_centroids.npy"),
                np.array(inner_hydroelastic_contact_result_centroids, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "inner_hydroelastic_contact_result_forces.npy"),
                np.array(inner_hydroelastic_contact_result_forces, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "inner_hydroelastic_contact_result_torques.npy"),
                np.array(inner_hydroelastic_contact_result_torques, dtype=object),
            )
            np.save(
                os.path.join(self._time_logs_dir_path, "inner_contact_result_times.npy"),
                np.array(inner_contact_result_times),
            )

    def save_data(self) -> None:
        # Meta data
        meta_data = {
            "logger_creation_timestamp": self._creation_timestamp,
            "logging_timestamp": str(datetime.datetime.now()),
            "manipuland_mass_estimated": self._manipuland_mass_estimated,
            "manipuland_inertia_estimated": self._manipuland_inertia_estimated,
            "time_taken_to_simulate_outer_s": self._outer_simulation_time,
            "time_taken_to_simulate_inner_s": self._inner_simulation_time,
        }
        meta_data.update(self._meta_data)
        with open(self._meta_data_file_path, "w") as f:
            yaml.dump(meta_data, f)

        # Experiment description
        with open(self._experiment_description_file_path, "w") as f:
            yaml.dump(self._experiment_description, f)

        # Camera data
        lengths = [
            len(self._camera_poses),
            len(self._intrinsics),
            len(self._images),
            len(self._depths),
            len(self._labels),
            len(self._masks),
        ]
        assert all(l == lengths[0] for l in lengths), f"All camera data must have the same length. Lengths: {lengths}"

        for i, (pose, intrinsics, image, depth, labels, masks) in enumerate(
            zip(self._camera_poses, self._intrinsics, self._images, self._depths, self._labels, self._masks)
        ):

            np.savetxt(os.path.join(self._camera_poses_dir_path, f"pose{i:04d}.txt"), pose)
            np.savetxt(os.path.join(self._intrinsics_dir_path, f"intrinsics{i:04d}.txt"), intrinsics)
            np.savetxt(os.path.join(self._depths_dir_path, f"depth{i:04d}.txt"), depth)

            image_pil = Image.fromarray(image)
            image_pil.save(os.path.join(self._images_dir_path, f"image{i:04d}.png"))

            mask_pil = None
            for label, mask in zip(labels, masks):
                if label == self._label_to_mask:
                    mask_pil = Image.fromarray(mask)
            if mask_pil is None:
                # Save black image
                mask_pil = Image.new("RGB", (image_pil.width, image_pil.height))
            mask_pil.save(os.path.join(self._masks_dir_path, f"mask{i:04d}.png"))

        self.save_manipuland_pose_logs()
        self.save_manipuland_contact_force_logs()
        self.save_contact_result_force_logs(self._manipuland_base_link_name)

        # Mesh data
        self.save_mesh_data()

        self.postprocess_data()
