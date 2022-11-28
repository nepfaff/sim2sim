"""sim2sim entrypoint with the table PID scene."""

#!/bin/python3
import os
import shutil
import argparse
import pathlib
from typing import List, Tuple

import numpy as np
import open3d as o3d
from pydrake.all import (
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    ProcessModelDirectives,
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    BasicVector,
    RigidTransform,
    DiagramBuilder,
    RollPitchYaw,
    PidController,
    SceneGraph,
)

from sim2sim.simulation import TablePIDSimulator
from sim2sim.logging import DynamicLogger
from sim2sim.util import get_parser, calc_mesh_inertia
from sim2sim.images import SphereImageGenerator
from sim2sim.mesh_processing import IdentityMeshProcessor

SCENE_DIRECTIVE = "../models/table_pid_scene_directive.yaml"
MANIPULAND_DIRECTIVE = "../models/table_pid_manipuland_directive.yaml"
MANIPULAND_BASE_LINK_NAME = "ycb_tomato_soup_can_base_link"
MANIPULANT_DEFAULT_POSE = RigidTransform(RollPitchYaw(-np.pi / 2.0, 0.0, 0.0), [0.0, 0.0, 0.6])  # X_WManipuland


class TableAngleSource(LeafSystem):
    def __init__(self, angle: float, no_command_time: float):
        """
        Commands zero for `no_command_time` and then commands `angle`.

        :param angle: The angle to command the table to.
        :param no_command_time: The time for which we command zero instead of `angle`.
        """
        LeafSystem.__init__(self)

        self._angle = angle
        self._start_time = None
        self._no_command_time = no_command_time

        self._control_output_port = self.DeclareVectorOutputPort("table_angle", BasicVector(2), self.CalcOutput)

    def CalcOutput(self, context, output):
        if context.get_time() < self._no_command_time:
            table_angle = 0.0
        else:
            table_angle = self._angle

        output.SetFromVector([table_angle, 0.0])


def create_env(
    args: argparse.Namespace,
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
    manipuland_pose: RigidTransform = MANIPULANT_DEFAULT_POSE,
) -> Tuple[DiagramBuilder, SceneGraph]:
    """Creates the table PID simulation environments without building it."""
    # Create plant
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, args.timestep)
    parser = get_parser(plant)
    for directive_path in directive_files:
        directive = LoadModelDirectives(directive_path)
        ProcessModelDirectives(directive, parser)
    for directive_str in directive_strs:
        directive = LoadModelDirectivesFromString(directive_str)
        ProcessModelDirectives(directive, parser)

    plant.SetDefaultFreeBodyPose(plant.GetBodyByName(MANIPULAND_BASE_LINK_NAME), manipuland_pose)
    plant.Finalize()

    # Table controller
    pid_controller = builder.AddSystem(PidController(kp=np.array([10.0]), ki=np.array([1.0]), kd=np.array([1.0])))
    pid_controller.set_name("pid_controller")

    # Now "wire up" the controller to the plant
    table_instance = plant.GetModelInstanceByName("table")
    builder.Connect(plant.get_state_output_port(table_instance), pid_controller.get_input_port_estimated_state())
    builder.Connect(pid_controller.get_output_port_control(), plant.get_actuation_input_port(table_instance))

    table_angle_source = builder.AddSystem(
        TableAngleSource(args.final_table_angle, no_command_time=args.no_command_time)
    )
    table_angle_source.set_name("table_angle_source")
    builder.Connect(table_angle_source.get_output_port(), pid_controller.get_input_port_desired_state())

    return builder, scene_graph


def create_processed_mesh_sdf_file(
    mass: float, inertia: np.ndarray, processed_mesh_file_path: str, tmp_folder: str
) -> str:
    """
    Creates and saves an SDF file for the processed mesh.

    :param mass: The object mass in kg.
    :param inertia: The moment of inertia matrix of shape (3,3).
    :param processed_mesh_file_path: The path to the processed mesh obj file.
    :param tmp_folder: The folder to write the sdf file to.
    :return procesed_mesh_sdf_path: The path to the SDF file.
    """
    procesed_mesh_sdf_str = f"""
        <?xml version="1.0"?>
        <sdf version="1.7">
            <model name="processed_manipuland_mesh">
                <link name="{MANIPULAND_BASE_LINK_NAME}">
                    <inertial>
                        <inertia>
                            <ixx>{inertia[0,0]}</ixx>
                            <ixy>{inertia[0,1]}</ixy>
                            <ixz>{inertia[0,2]}</ixz>
                            <iyy>{inertia[1,1]}</iyy>
                            <iyz>{inertia[1,2]}</iyz>
                            <izz>{inertia[2,2]}</izz>
                        </inertia>
                        <mass>{mass}</mass>
                    </inertial>

                    <visual name="visual">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{processed_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
                    </visual>
                    <collision name="collision">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{processed_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
                        <drake:proximity_properties>
                            <drake:rigid_hydroelastic/>
                        </drake:proximity_properties>
                    </collision>
                </link>
            </model>
        </sdf>
    """
    procesed_mesh_sdf_path = os.path.join(tmp_folder, "processed_mesh.sdf")
    with open(procesed_mesh_sdf_path, "w") as f:
        f.write(procesed_mesh_sdf_str)

    return procesed_mesh_sdf_path


def create_processed_mesh_directive_str(
    mass: float, inertia: np.ndarray, processed_mesh_file_path: str, tmp_folder: str
) -> str:
    """
    Creates a directive for the processed mesh.

    :param mass: The object mass in kg.
    :param inertia: The moment of inertia matrix of shape (3,3).
    :param processed_mesh_file_path: The path to the processed mesh obj file.
    :param tmp_folder: The folder to write the sdf file to.
    :return processed_mesh_directive_str: The directive string for the processed mesh.
    """
    procesed_mesh_sdf_path = create_processed_mesh_sdf_file(mass, inertia, processed_mesh_file_path, tmp_folder)
    processed_mesh_directive = f"""
        directives:
        - add_model:
            name: processed_manipuland_mesh
            file: package://sim2sim/{procesed_mesh_sdf_path}
    """
    return processed_mesh_directive


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--sim_duration",
        type=float,
        default=3.0,
        required=False,
        help="The simulation duration in seconds.",
    )
    argument_parser.add_argument(
        "--timestep",
        type=float,
        default=0.0001,
        required=False,
        help="The timestep to use in seconds.",
    )
    argument_parser.add_argument(
        "--final_table_angle",
        type=float,
        default=np.pi / 4,
        required=False,
        help="The final table angle in radians.",
    )
    argument_parser.add_argument(
        "--no_command_time",
        type=float,
        default=1.0,
        required=False,
        help="The time before starting the table control in seconds.",
    )
    argument_parser.add_argument(
        "--realtime_rate",
        type=float,
        default=1.0,
        required=False,
        help="The simulation realtime rate.",
    )
    argument_parser.add_argument(
        "--html",
        type=str,
        required=False,
        help="Path to save the meshcat html to. The file should end with .html.",
    )
    argument_parser.add_argument(
        "--logging_path",
        type=str,
        default="test_logging_path",
        required=False,
        help="The path to log the data to.",
    )
    argument_parser.add_argument(
        "--kProximity", action="store_true", help="Whether to visualize kProximity or kIllustration."
    )
    # TODO: Use this argument to turn the hydroelastic engine on or off
    # argument_parser.add_argument(
    #     "--use_hydroelastic", action="store_true", help="Whether to use the Hydroelastic contact model."
    # )
    argument_parser.add_argument("--contact_viz", action="store_true", help="Whether to visualize the contact forces.")
    args = argument_parser.parse_args()
    return args


def main():
    args = parse_args()

    scene_directive = os.path.join(pathlib.Path(__file__).parent.resolve(), SCENE_DIRECTIVE)
    manipuland_directive = os.path.join(pathlib.Path(__file__).parent.resolve(), MANIPULAND_DIRECTIVE)

    # Label 4 is the Tomato Soup Can in this simulation setup
    logger = DynamicLogger(logging_frequency_hz=0.001, logging_path=args.logging_path, label_to_mask=4)

    # Create folder for temporary files
    tmp_folder = os.path.join(args.logging_path, "tmp")
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    builder_outer, scene_graph_outer = create_env(directive_files=[scene_directive, manipuland_directive], args=args)

    # Create a new version of the scene for generating camera data
    builder_camera, scene_graph_camera = create_env(directive_files=[scene_directive, manipuland_directive], args=args)
    image_generator = SphereImageGenerator(
        builder=builder_camera,
        scene_graph=scene_graph_camera,
        logger=logger,
        simulate_time=args.no_command_time,
        look_at_point=MANIPULANT_DEFAULT_POSE.translation(),
        z_distances=[0.02, 0.2, 0.3],
        radii=[0.4, 0.3, 0.3],
        num_poses=[30, 25, 15],
    )

    images, intrinsics, extrinsics, depths, labels, masks = image_generator.generate_images()
    print("Finished generating images.")

    # TODO: Replace this with a call to identity inverse graphics
    raw_mesh = o3d.io.read_triangle_mesh("./data/ycb_tomato_soup_can/ycb_tomato_soup_can.obj")
    raw_mesh_pose = RigidTransform(RollPitchYaw(0.0, 0.0, 0.0), [0.0, 0.0, 0.6])

    mesh_processor = IdentityMeshProcessor()
    processed_mesh = mesh_processor.process_mesh(raw_mesh)

    # Compute mesh inertia and mass assuming constant density of water
    mass, inertia = calc_mesh_inertia(processed_mesh)

    # Save mesh data to create SDF files that can be added to a new simulation environment
    logger.log(raw_mesh=raw_mesh, processed_mesh=processed_mesh)
    _, processed_mesh_file_path = logger.save_mesh_data()
    processed_mesh_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "..", processed_mesh_file_path)

    # Create a directive for processed_mesh manipuland
    processed_mesh_directive = create_processed_mesh_directive_str(mass, inertia, processed_mesh_file_path, tmp_folder)

    builder_inner, scene_graph_inner = create_env(
        directive_files=[scene_directive],
        directive_strs=[processed_mesh_directive],
        args=args,
        manipuland_pose=raw_mesh_pose,
    )

    simulator = TablePIDSimulator(builder_outer, scene_graph_outer, builder_inner, scene_graph_inner, logger)
    simulator.simulate(args.sim_duration)

    print("Saving data.")
    logger.save_data()

    # Clean up temporary files
    shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    main()
