import os
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
    Sphere,
    SpatialInertia,
    UnitInertia,
    PrismaticJoint,
    InverseDynamicsController,
)
from manipulation.scenarios import AddShape

from sim2sim.simulation import BasicSimulator, BasicInnerOnlySimulator, SpherePushingSimulator
from sim2sim.logging import DynamicLogger
from sim2sim.util import (
    get_parser,
    create_processed_mesh_directive_str,
    create_processed_mesh_primitive_directive_str,
    SphereStateSource,
)
from sim2sim.images import SphereImageGenerator, NoneImageGenerator
from sim2sim.inverse_graphics import IdentityInverseGraphics
from sim2sim.mesh_processing import (
    IdentityMeshProcessor,
    QuadricDecimationMeshProcessor,
    SphereMeshProcessor,
    GMMMeshProcessor,
    ConvexDecompMeshProcessor,
    CoACDMeshProcessor,
    FuzzyMetaballMeshProcessor,
)
from sim2sim.physical_property_estimator import WaterDensityPhysicalPropertyEstimator, GTPhysicalPropertyEstimator

SPHERE_STARTING_POSITION = [0.3, 0.0, 0.1]
SCENE_DIRECTIVE = "../../models/random_force/random_force_directive.yaml"

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
    "GMMMeshProcessor": GMMMeshProcessor,
    "ConvexDecompMeshProcessor": ConvexDecompMeshProcessor,
    "CoACDMeshProcessor": CoACDMeshProcessor,
    "FuzzyMetaballMeshProcessor": FuzzyMetaballMeshProcessor,
}
PHYSICAL_PROPERTY_ESTIMATOR = {
    "WaterDensityPhysicalPropertyEstimator": WaterDensityPhysicalPropertyEstimator,
    "GTPhysicalPropertyEstimator": GTPhysicalPropertyEstimator,
}
SIMULATORS = {
    "BasicSimulator": BasicSimulator,
    "BasicInnerOnlySimulator": BasicInnerOnlySimulator,
    "SpherePushingSimulator": SpherePushingSimulator,
}


def add_sphere(plant: MultibodyPlant, radius: float = 0.05, position: List[float] = [0.0, 0.0, 0.0]) -> None:
    sphere = AddShape(plant, Sphere(radius), "sphere", color=[0.9, 0.5, 0.5, 1.0])
    _ = plant.AddRigidBody("false_body1", sphere, SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0)))
    sphere_x = plant.AddJoint(
        PrismaticJoint("sphere_x", plant.world_frame(), plant.GetFrameByName("false_body1"), [1, 0, 0], -10.0, 10.0)
    )
    sphere_x.set_default_translation(position[0])
    plant.AddJointActuator("sphere_x", sphere_x)
    _ = plant.AddRigidBody("false_body2", sphere, SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0)))
    sphere_y = plant.AddJoint(
        PrismaticJoint(
            "sphere_y", plant.GetFrameByName("false_body1"), plant.GetFrameByName("false_body2"), [0, 1, 0], -10.0, 10.0
        )
    )
    sphere_y.set_default_translation(position[1])
    plant.AddJointActuator("sphere_y", sphere_y)
    sphere_z = plant.AddJoint(
        PrismaticJoint(
            "sphere_z", plant.GetFrameByName("false_body2"), plant.GetFrameByName("sphere"), [0, 0, 1], -10.0, 10.0
        )
    )
    sphere_z.set_default_translation(position[2])
    plant.AddJointActuator("sphere_z", sphere_z)


def create_env(
    env_params: dict,
    timestep: float,
    manipuland_base_link_name: str,
    manipuland_pose: RigidTransform,
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """Creates the sphere pushing simulation environment without building it."""

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

    add_sphere(plant, position=SPHERE_STARTING_POSITION)

    # Sphere state source
    sphere_state_source = builder.AddSystem(SphereStateSource(SPHERE_STARTING_POSITION))
    sphere_state_source.set_name("sphere_state_source")

    # Sphere controller
    sphere_controller_plant = MultibodyPlant(time_step=timestep)
    sphere_controller_plant.set_name("sphere_controller_plant")
    add_sphere(sphere_controller_plant)
    sphere_controller_plant.Finalize()
    sphere_inverse_dynamics_controller = builder.AddSystem(
        InverseDynamicsController(
            sphere_controller_plant,
            kp=[100.0] * 3,
            ki=[1.0] * 3,
            kd=[20.0] * 3,
            has_reference_acceleration=False,
        )
    )
    sphere_inverse_dynamics_controller.set_name("sphere_inverse_dynamics_controller")

    plant.SetDefaultFreeBodyPose(plant.GetBodyByName(manipuland_base_link_name), manipuland_pose)
    plant.Finalize()

    # Connect sphere state source and controller to plant
    sphere_instance = plant.GetModelInstanceByName("sphere")
    builder.Connect(
        plant.get_state_output_port(sphere_instance),
        sphere_inverse_dynamics_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        sphere_inverse_dynamics_controller.get_output_port_control(), plant.get_actuation_input_port(sphere_instance)
    )
    builder.Connect(
        sphere_state_source.get_output_port(), sphere_inverse_dynamics_controller.get_input_port_desired_state()
    )

    return builder, scene_graph, plant


def run_sphere_pushing(
    logging_path: str,
    params: dict,
    sim_duration: float,
    timestep: float,
    logging_frequency_hz: float,
    manipuland_directive: str,
    manipuland_base_link_name: str,
    manipuland_default_pose: str,
    save_raw_mesh: bool,
    hydroelastic_manipuland: bool,
):
    """
    Experiment entrypoint for the sphere pushing scene.

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
    :param hydroelastic_manipuland: Whether to use hydroelastic or point contact for the inner manipuland.
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
    is_primitive, processed_mesh, processed_mesh_piece, primitive_info = mesh_processor.process_mesh(raw_mesh)
    print("Finished mesh processing.")

    # Compute mesh inertia and mass assuming constant density of water
    physical_property_estimator_class = PHYSICAL_PROPERTY_ESTIMATOR[params["physical_property_estimator"]["class"]]
    physical_porperty_estimator = physical_property_estimator_class(
        **(
            params["physical_property_estimator"]["args"]
            if params["physical_property_estimator"]["args"] is not None
            else {}
        ),
    )
    mass, inertia = physical_porperty_estimator.estimate_physical_properties(processed_mesh)
    print("Finished estimating physical properties.")
    logger.log_manipuland_estimated_physics(manipuland_mass_estimated=mass, manipuland_inertia_estimated=inertia)

    # Save mesh data to create SDF files that can be added to a new simulation environment
    if save_raw_mesh:
        logger.log(raw_mesh=raw_mesh)
    logger.log(processed_mesh=processed_mesh, processed_mesh_piece=processed_mesh_piece)
    _, processed_mesh_file_path = logger.save_mesh_data()
    processed_mesh_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "../..", processed_mesh_file_path)

    # Create a directive for processed_mesh manipuland
    if is_primitive:
        processed_mesh_directive = create_processed_mesh_primitive_directive_str(
            primitive_info,
            mass,
            inertia,
            logger._mesh_dir_path,
            params["env"]["obj_name"],
            manipuland_base_link_name,
            hydroelastic=hydroelastic_manipuland,
        )
    else:
        processed_mesh_directive = create_processed_mesh_directive_str(
            mass,
            inertia,
            processed_mesh_file_path,
            logger._mesh_dir_path,
            params["env"]["obj_name"],
            manipuland_base_link_name,
            hydroelastic=hydroelastic_manipuland,
        )

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
