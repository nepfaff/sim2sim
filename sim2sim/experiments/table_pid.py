import os
import shutil
import pathlib
from typing import List, Tuple

import numpy as np
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
    MultibodyPlant,
)

from sim2sim.simulation import TablePIDSimulator
from sim2sim.logging import DynamicLogger
from sim2sim.util import get_parser, calc_mesh_inertia
from sim2sim.images import SphereImageGenerator
from sim2sim.inverse_graphics import IdentityInverseGraphics
from sim2sim.mesh_processing import IdentityMeshProcessor

SCENE_DIRECTIVE = "../../models/table_pid_scene_directive.yaml"

# TODO: Allow specifying manipulant with experiment yaml file
MANIPULAND_DIRECTIVE = "../../models/table_pid_manipuland_directive.yaml"
MANIPULAND_NAME = "ycb_tomato_soup_can"
MANIPULAND_BASE_LINK_NAME = "ycb_tomato_soup_can_base_link"
MANIPULANT_DEFAULT_POSE = RigidTransform(RollPitchYaw(-np.pi / 2.0, 0.0, 0.0), [0.0, 0.0, 0.6])  # X_WManipuland

# TODO: Add type info using base classes
LOGGERS = {
    "DynamicLogger": DynamicLogger,
}
IMAGE_GENERATORS = {
    "SphereImageGenerator": SphereImageGenerator,
}
INVERSE_GRAPHICS = {
    "IdentityInverseGraphics": IdentityInverseGraphics,
}
MESH_PROCESSORS = {
    "IdentityMeshProcessor": IdentityMeshProcessor,
}
SIMULATORS = {
    "TablePIDSimulator": TablePIDSimulator,
}


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
    timestep: float,
    final_table_angle: float,
    no_command_time: float,
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
    manipuland_pose: RigidTransform = MANIPULANT_DEFAULT_POSE,
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """Creates the table PID simulation environments without building it."""
    # Create plant
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, timestep)
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

    table_angle_source = builder.AddSystem(TableAngleSource(final_table_angle, no_command_time=no_command_time))
    table_angle_source.set_name("table_angle_source")
    builder.Connect(table_angle_source.get_output_port(), pid_controller.get_input_port_desired_state())

    return builder, scene_graph, plant


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
            name: ycb_tomato_soup_can
            file: package://sim2sim/{procesed_mesh_sdf_path}
    """
    return processed_mesh_directive


def run_table_pid(
    logging_path: str,
    params: dict,
    sim_duration: float,
    timestep: float,
    final_table_angle: float,
    no_command_time: float,
    logging_frequency_hz: float,
):
    """
    Experiment entrypoint for the table PID scene.

    :param logging_path: The path to log the data to.
    :param params: The experiment yaml file dict.
    :param sim_duration: The simulation duration in seconds.
    :param timestep: The timestep to use in seconds.
    :param final_table_angle: The final table angle in radians.
    :param no_command_time: The time before starting the table control in seconds.
    :param logging_frequency_hz: The dynamics logging frequency.
    """

    scene_directive = os.path.join(pathlib.Path(__file__).parent.resolve(), SCENE_DIRECTIVE)
    manipuland_directive = os.path.join(pathlib.Path(__file__).parent.resolve(), MANIPULAND_DIRECTIVE)

    logger_class = LOGGERS[params["logger"]["class"]]
    logger = logger_class(
        logging_frequency_hz=logging_frequency_hz,
        logging_path=logging_path,
        **(params["logger"]["args"] if params["logger"]["args"] is not None else {}),
    )

    # Create folder for temporary files
    tmp_folder = os.path.join(logging_path, "tmp")
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    builder_outer, scene_graph_outer, outer_plant = create_env(
        timestep,
        final_table_angle,
        no_command_time,
        directive_files=[scene_directive, manipuland_directive],
    )

    # Create a new version of the scene for generating camera data
    builder_camera, scene_graph_camera, _ = create_env(
        timestep, final_table_angle, no_command_time, directive_files=[scene_directive, manipuland_directive]
    )
    image_generator_class = IMAGE_GENERATORS[params["image_generator"]["class"]]
    image_generator = image_generator_class(
        builder=builder_camera,
        scene_graph=scene_graph_camera,
        logger=logger,
        **(params["image_generator"]["args"] if params["image_generator"]["args"] is not None else {}),
    )

    images, intrinsics, extrinsics, depths, labels, masks = image_generator.generate_images()
    print("Finished generating images.")

    inverse_graphics_class = INVERSE_GRAPHICS[params["inverse_graphics"]["class"]]
    inverse_graphics = inverse_graphics_class(
        **(params["inverse_graphics"]["args"] if params["inverse_graphics"]["args"] is not None else {}),
        images=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth=depths,
        labels=labels,
        masks=masks,
    )
    raw_mesh, raw_mesh_pose = inverse_graphics.run()
    print("Finished running inverse graphics.")

    mesh_processor_class = MESH_PROCESSORS[params["mesh_processor"]["class"]]
    mesh_processor = mesh_processor_class(
        **(params["mesh_processor"]["args"] if params["mesh_processor"]["args"] is not None else {}),
    )
    processed_mesh = mesh_processor.process_mesh(raw_mesh)
    print("Finished mesh processing.")

    # Compute mesh inertia and mass assuming constant density of water
    mass, inertia = calc_mesh_inertia(processed_mesh)

    # Save mesh data to create SDF files that can be added to a new simulation environment
    logger.log(raw_mesh=raw_mesh, processed_mesh=processed_mesh)
    _, processed_mesh_file_path = logger.save_mesh_data()
    processed_mesh_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "../..", processed_mesh_file_path)

    # Create a directive for processed_mesh manipuland
    processed_mesh_directive = create_processed_mesh_directive_str(mass, inertia, processed_mesh_file_path, tmp_folder)

    builder_inner, scene_graph_inner, inner_plant = create_env(
        timestep,
        final_table_angle,
        no_command_time,
        directive_files=[scene_directive],
        directive_strs=[processed_mesh_directive],
        manipuland_pose=RigidTransform(RollPitchYaw(*raw_mesh_pose[:3]), raw_mesh_pose[3:]),
    )

    logger.add_plants(outer_plant, inner_plant)

    simulator_class = SIMULATORS[params["simulator"]["class"]]
    simulator = simulator_class(
        outer_builder=builder_outer,
        outer_scene_graph=scene_graph_outer,
        inner_builder=builder_inner,
        inner_scene_graph=scene_graph_inner,
        logger=logger,
        **(params["simulator"]["args"] if params["simulator"]["args"] is not None else {}),
    )
    simulator.simulate(sim_duration)
    print("Finished simulating.")

    logger.save_data()
    print("Finished saving data.")

    # Clean up temporary files
    shutil.rmtree(tmp_folder)
