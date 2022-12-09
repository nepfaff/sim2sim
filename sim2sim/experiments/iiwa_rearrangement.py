import os
import shutil
import pathlib
from typing import List, Tuple

import numpy as np
from pydrake.all import (
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    ProcessModelDirectives,
    AddMultibodyPlant,
    RigidTransform,
    DiagramBuilder,
    RollPitchYaw,
    SceneGraph,
    MultibodyPlant,
    Demultiplexer,
    MultibodyPlantConfig,
)

from sim2sim.logging import DynamicLogger
from sim2sim.util import (
    get_parser,
    calc_mesh_inertia,
    create_processed_mesh_directive_str,
    add_iiwa_system,
    add_cameras,
    add_wsg_system,
    IIWAJointTrajectorySource,
    WSGCommandSource,
    IIWAControlModeSource,
)
from sim2sim.images import SphereImageGenerator, IIWAWristSphereImageGenerator
from sim2sim.inverse_graphics import IdentityInverseGraphics
from sim2sim.mesh_processing import IdentityMeshProcessor, QuadricDecimationMeshProcessor
from sim2sim.simulation import BasicSimulator, IIWARearrangementSimulator, IIWASlideInHoleSimulator

SCENE_DIRECTIVE = "../../models/iiwa_rearrangement/iiwa_rearrangement_scene_directive.yaml"
IIWA_Q_NOMINAL = np.array([1.5, -0.4, 0.0, -1.75, 0.0, 1.5, 0.0])  # iiwa joint angles in radians

# TODO: Allow specifying manipulant with experiment yaml file
MANIPULAND_DIRECTIVE = "../../models/iiwa_rearrangement/iiwa_rearrangement_manipuland_directive.yaml"
MANIPULAND_NAME = "ycb_tomato_soup_can"
MANIPULAND_BASE_LINK_NAME = "ycb_tomato_soup_can_base_link"
MANIPULANT_DEFAULT_POSE = RigidTransform(RollPitchYaw(-np.pi / 2.0, 0.0, 0.0), [0.0, 0.5, 0.050450])  # X_WManipuland

LOGGERS = {
    "DynamicLogger": DynamicLogger,
}
IMAGE_GENERATORS = {
    "SphereImageGenerator": SphereImageGenerator,
    "IIWAWristSphereImageGenerator": IIWAWristSphereImageGenerator,
}
INVERSE_GRAPHICS = {
    "IdentityInverseGraphics": IdentityInverseGraphics,
}
MESH_PROCESSORS = {
    "IdentityMeshProcessor": IdentityMeshProcessor,
    "QuadricDecimationMeshProcessor": QuadricDecimationMeshProcessor,
}
SIMULATORS = {
    "BasicSimulator": BasicSimulator,
    "IIWARearrangementSimulator": IIWARearrangementSimulator,
    "IIWASlideInHoleSimulator": IIWASlideInHoleSimulator,
}


def create_env(
    timestep: float,
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
    manipuland_pose: RigidTransform = MANIPULANT_DEFAULT_POSE,
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """Creates the iiwa rearrangement simulation environments without building it."""

    # Create plant
    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(
        time_step=timestep,
        contact_model="hydroelastic_with_fallback",  # "point"
        discrete_contact_solver="sap",
    )
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    plant.set_name("plant")
    parser = get_parser(plant)
    for directive_path in directive_files:
        directive = LoadModelDirectives(directive_path)
        ProcessModelDirectives(directive, parser)
    for directive_str in directive_strs:
        directive = LoadModelDirectivesFromString(directive_str)
        ProcessModelDirectives(directive, parser)

    plant.SetDefaultFreeBodyPose(plant.GetBodyByName(MANIPULAND_BASE_LINK_NAME), manipuland_pose)
    plant.Finalize()

    # Add iiwa controller
    (
        iiwa_controller_plant,
        iiwa_control_mode_input,
        iiwa_position_input,
        iiwa_position_commanded_output,
        iiwa_state_estimated_output,
        iiwa_position_measured_output,
        iiwa_velocity_estimated_output,
        iiwa_feedforward_torque_input,
        iiwa_torque_commanded_output,
        iiwa_torque_external_output,
    ) = add_iiwa_system(
        builder=builder, plant=plant, iiwa_instance_idx=plant.GetModelInstanceByName("iiwa"), iiwa_time_step=timestep
    )
    iiwa_control_mode_source = builder.AddSystem(IIWAControlModeSource())
    iiwa_control_mode_source.set_name("iiwa_control_mode_source")
    builder.Connect(iiwa_control_mode_source.GetOutputPort("iiwa_control_mode"), iiwa_control_mode_input)

    # Add wsg controller
    (
        wsg_position_input,
        wsg_force_limit_input,
        wsg_state_measured_output,
        wsg_force_measured_output,
    ) = add_wsg_system(builder=builder, plant=plant, wsg_instance_idx=plant.GetModelInstanceByName("wsg"))

    add_cameras(
        builder=builder,
        plant=plant,
        scene_graph=scene_graph,
        width=1920,
        height=1440,
        fov_y=np.pi / 4.0,
    )

    # Add iiwa joint trajectory source
    iiwa_joint_trajectory_source = builder.AddSystem(
        IIWAJointTrajectorySource(
            plant=iiwa_controller_plant,
            q_nominal=IIWA_Q_NOMINAL,
        )
    )
    iiwa_joint_trajectory_source.set_name("iiwa_joint_trajectory_source")
    demux = builder.AddSystem(Demultiplexer(14, 7))  # Assume 7 iiwa joint positions
    builder.Connect(iiwa_joint_trajectory_source.get_output_port(), demux.get_input_port())
    builder.Connect(demux.get_output_port(0), iiwa_position_input)

    # Add wsg position source
    wsg_command_source = builder.AddSystem(WSGCommandSource(initial_pos=0.1))
    wsg_command_source.set_name("wsg_command_source")
    builder.Connect(wsg_command_source.get_output_port(), wsg_position_input)

    return builder, scene_graph, plant


def run_iiwa_rearrangement(
    logging_path: str,
    params: dict,
    sim_duration: float,
    timestep: float,
    logging_frequency_hz: float,
):
    """
    Experiment entrypoint for the iiwa rearrangement scene.

    :param logging_path: The path to log the data to.
    :param params: The experiment yaml file dict.
    :param sim_duration: The simulation duration in seconds.
    :param timestep: The timestep to use in seconds.
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
        directive_files=[scene_directive, manipuland_directive],
    )

    # Create a new version of the scene for generating camera data
    builder_camera, scene_graph_camera, _ = create_env(
        timestep, directive_files=[scene_directive, manipuland_directive]
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
    processed_mesh_directive = create_processed_mesh_directive_str(
        mass, inertia, processed_mesh_file_path, tmp_folder, "ycb_tomato_soup_can", MANIPULAND_BASE_LINK_NAME
    )

    builder_inner, scene_graph_inner, inner_plant = create_env(
        timestep,
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
