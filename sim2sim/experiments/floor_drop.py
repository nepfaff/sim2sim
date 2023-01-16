import os
import shutil
import pathlib
from typing import List, Tuple

from pydrake.all import (
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    ProcessModelDirectives,
    RigidTransform,
    DiagramBuilder,
    RollPitchYaw,
    SceneGraph,
    MultibodyPlant,
    MultibodyPlantConfig,
    AddMultibodyPlant,
)

from sim2sim.simulation import BasicSimulator, BasicInnerOnlySimulator
from sim2sim.logging import DynamicLogger
from sim2sim.util import get_parser, calc_mesh_inertia, create_processed_mesh_directive_str
from sim2sim.images import SphereImageGenerator, NoneImageGenerator
from sim2sim.inverse_graphics import IdentityInverseGraphics
from sim2sim.mesh_processing import (
    IdentityMeshProcessor,
    QuadricDecimationMeshProcessor,
    SphereMeshProcessor,
    MetaBallMeshProcessor,
    ConvexDecompMeshProcessor,
    CoACDMeshProcessor,
)

SCENE_DIRECTIVE = "../../models/floor_drop/floor_drop_directive.yaml"

# TODO: Add type info using base classes
LOGGERS = {
    "DynamicLogger": DynamicLogger,
}
IMAGE_GENERATORS = {
    "NoneImageGenerator": NoneImageGenerator,
    "SphereImageGenerator": SphereImageGenerator,
}
INVERSE_GRAPHICS = {
    "IdentityInverseGraphics": IdentityInverseGraphics,
}
MESH_PROCESSORS = {
    "IdentityMeshProcessor": IdentityMeshProcessor,
    "QuadricDecimationMeshProcessor": QuadricDecimationMeshProcessor,
    "SphereMeshProcessor": SphereMeshProcessor,
    "MetaBallMeshProcessor": MetaBallMeshProcessor,
    "ConvexDecompMeshProcessor": ConvexDecompMeshProcessor,
    "CoACDMeshProcessor": CoACDMeshProcessor,
}
SIMULATORS = {
    "BasicSimulator": BasicSimulator,
    "BasicInnerOnlySimulator": BasicInnerOnlySimulator,
}


def create_env(
    env_params: dict,
    timestep: float,
    manipuland_base_link_name: str,
    manipuland_pose: RigidTransform,
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """Creates the floor drop simulation environments without building it."""

    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(
        time_step=timestep,
        contact_model=env_params["contact_model"],
        discrete_contact_solver=env_params["solver"],
    )
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    parser = get_parser(plant)
    for directive_path in directive_files:
        directive = LoadModelDirectives(directive_path)
        ProcessModelDirectives(directive, parser)
    for directive_str in directive_strs:
        directive = LoadModelDirectivesFromString(directive_str)
        ProcessModelDirectives(directive, parser)

    plant.SetDefaultFreeBodyPose(plant.GetBodyByName(manipuland_base_link_name), manipuland_pose)
    plant.Finalize()

    return builder, scene_graph, plant


def run_floor_drop(
    logging_path: str,
    params: dict,
    sim_duration: float,
    timestep: float,
    logging_frequency_hz: float,
    manipuland_directive: str,
    manipuland_base_link_name: str,
    manipuland_default_pose: str,
    save_raw_mesh: bool,
):
    """
    Experiment entrypoint for the floor drop scene.

    :param logging_path: The path to log the data to.
    :param params: The experiment yaml file dict.
    :param sim_duration: The simulation duration in seconds.
    :param timestep: The timestep to use in seconds.
    :param logging_frequency_hz: The dynamics logging frequency.
    :param manipuland_directive: The file path of the outer manipuland directive. The path should be relative to this
        script.
    :param manipuland_base_link_name: The base link name of the outer manipuland.
    :param manipuland_default_pose: The default pose of the outer manipuland of form [roll, pitch, yaw, x, y, z].
    :param save_raw_mesh: Whether to save the raw mesh from inverse graphics.
    """

    scene_directive = os.path.join(pathlib.Path(__file__).parent.resolve(), SCENE_DIRECTIVE)
    manipuland_directive_path = os.path.join(pathlib.Path(__file__).parent.resolve(), manipuland_directive)

    logger_class = LOGGERS[params["logger"]["class"]]
    logger = logger_class(
        logging_frequency_hz=logging_frequency_hz,
        logging_path=logging_path,
        **(params["logger"]["args"] if params["logger"]["args"] is not None else {}),
    )
    logger.log(experiment_description=params)

    # Create folder for temporary files
    tmp_folder = os.path.join(logging_path, "tmp")
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    manipuland_default_pose_transform = RigidTransform(
        RollPitchYaw(*manipuland_default_pose[:3]), manipuland_default_pose[3:]
    )
    builder_outer, scene_graph_outer, outer_plant = create_env(
        env_params=params["env"],
        timestep=timestep,
        manipuland_base_link_name=manipuland_base_link_name,
        manipuland_pose=manipuland_default_pose_transform,
        directive_files=[scene_directive, manipuland_directive_path],
    )

    # Create a new version of the scene for generating camera data
    builder_camera, scene_graph_camera, _ = create_env(
        timestep=timestep,
        env_params=params["env"],
        manipuland_base_link_name=manipuland_base_link_name,
        manipuland_pose=manipuland_default_pose_transform,
        directive_files=[scene_directive, manipuland_directive_path],
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
    # TODO: Log 'raw_mesh_pose' and 'manipuland_default_pose_transform' as meta-data
    print("Finished running inverse graphics.")

    mesh_processor_class = MESH_PROCESSORS[params["mesh_processor"]["class"]]
    mesh_processor = mesh_processor_class(
        logger=logger,
        **(params["mesh_processor"]["args"] if params["mesh_processor"]["args"] is not None else {}),
    )
    processed_mesh, processed_mesh_piece = mesh_processor.process_mesh(raw_mesh)
    print("Finished mesh processing.")

    # Compute mesh inertia and mass assuming constant density of water
    mass, inertia = calc_mesh_inertia(raw_mesh)  # processed_mesh
    logger.log_manipuland_estimated_physics(manipuland_mass_estimated=mass, manipuland_inertia_estimated=inertia)

    # Save mesh data to create SDF files that can be added to a new simulation environment
    if save_raw_mesh:
        logger.log(raw_mesh=raw_mesh)
    logger.log(processed_mesh=processed_mesh, processed_mesh_piece=processed_mesh_piece)
    _, processed_mesh_file_path = logger.save_mesh_data()
    processed_mesh_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "../..", processed_mesh_file_path)

    # Create a directive for processed_mesh manipuland
    processed_mesh_directive = create_processed_mesh_directive_str(
        mass, inertia, processed_mesh_file_path, tmp_folder, params["env"]["obj_name"], manipuland_base_link_name
    )

    #
    builder_inner, scene_graph_inner, inner_plant = create_env(
        timestep=timestep,
        env_params=params["env"],
        manipuland_base_link_name=manipuland_base_link_name,
        directive_files=[scene_directive],
        directive_strs=[processed_mesh_directive],
        manipuland_pose=RigidTransform(RollPitchYaw(*raw_mesh_pose[:3]), raw_mesh_pose[3:]),
    )

    logger.add_plants(outer_plant, inner_plant)
    logger.add_scene_graphs(scene_graph_outer, scene_graph_inner)

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
