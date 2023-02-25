import os
import pathlib
from typing import List, Tuple, Dict

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
    AddCompliantHydroelasticProperties,
    ProximityProperties,
    RoleAssign,
)
from manipulation.scenarios import AddShape

from sim2sim.simulation import BasicSimulator, SpherePushingSimulator
from sim2sim.logging import SpherePushingLogger
from sim2sim.util import (
    get_parser,
    create_processed_mesh_directive_str,
    create_processed_mesh_primitive_directive_str,
    SphereStateSource,
    copy_object_proximity_properties,
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
    IdentityPrimitiveMeshProcessor,
)
from sim2sim.physical_property_estimator import WaterDensityPhysicalPropertyEstimator, GTPhysicalPropertyEstimator

SCENE_DIRECTIVE = "../../models/random_force/random_force_directive.yaml"

# TODO: Add type info using base classes
LOGGERS = {
    "SpherePushingLogger": SpherePushingLogger,
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
    "IdentityPrimitiveMeshProcessor": IdentityPrimitiveMeshProcessor,
}
PHYSICAL_PROPERTY_ESTIMATOR = {
    "WaterDensityPhysicalPropertyEstimator": WaterDensityPhysicalPropertyEstimator,
    "GTPhysicalPropertyEstimator": GTPhysicalPropertyEstimator,
}
SIMULATORS = {
    "BasicSimulator": BasicSimulator,
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
    sphere_starting_position: List[float],
    sphere_pid_gains: Dict[str, float],
    hydroelastic_manipuland: bool,
    sphere_radius: float,
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

    add_sphere(plant, radius=sphere_radius, position=sphere_starting_position)
    if hydroelastic_manipuland:
        # Make sphere complient hydroelastic
        sphere = plant.GetBodyByName("sphere")
        new_proximity_properties = ProximityProperties()
        # NOTE: Setting hydroelastic properties becomes slow as the resolution hint decreases
        AddCompliantHydroelasticProperties(
            resolution_hint=0.01, hydroelastic_modulus=1e8, properties=new_proximity_properties
        )
        geometry_ids = plant.GetCollisionGeometriesForBody(sphere)
        for geometry_id in geometry_ids:
            inspector = scene_graph.model_inspector()
            const_proximity_properties = inspector.GetProximityProperties(geometry_id)
            copy_object_proximity_properties(const_proximity_properties, new_proximity_properties)
            scene_graph.AssignRole(plant.get_source_id(), geometry_id, new_proximity_properties, RoleAssign.kReplace)

    # Sphere state source
    sphere_state_source = builder.AddSystem(SphereStateSource(sphere_starting_position))
    sphere_state_source.set_name("sphere_state_source")

    # Sphere controller
    sphere_controller_plant = MultibodyPlant(time_step=timestep)
    sphere_controller_plant.set_name("sphere_controller_plant")
    add_sphere(sphere_controller_plant, radius=sphere_radius)
    sphere_controller_plant.Finalize()
    sphere_inverse_dynamics_controller = builder.AddSystem(
        InverseDynamicsController(
            sphere_controller_plant,
            kp=[sphere_pid_gains["kp"]] * 3,
            ki=[sphere_pid_gains["ki"]] * 3,
            kd=[sphere_pid_gains["kd"]] * 3,
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


def run_pipeline(
    params: dict,
    logger: SpherePushingLogger,
    timestep: float,
    manipuland_base_link_name: str,
    manipuland_default_pose: RigidTransform,
    save_raw_mesh: bool,
    hydroelastic_manipuland: bool,
    sphere_starting_position: List[float],
    sphere_pid_gains: Dict[str, float],
    sphere_radius: float,
    scene_directive_path: str,
    manipuland_directive_path: str,
    prefix: str = "",
):
    """
    Runs the sim2sim pipeline of camera data generation, mesh generation, mesh processing, and physical property
    estimation.

    :param prefix: The prefix of the pipeline components in `params`.
    """
    prefix = prefix + "_" if prefix else ""

    # Create a new version of the scene for generating camera data
    camera_builder, camera_scene_graph, _ = create_env(
        env_params=params["env"],
        timestep=timestep,
        manipuland_base_link_name=manipuland_base_link_name,
        manipuland_pose=manipuland_default_pose,
        sphere_starting_position=sphere_starting_position,
        sphere_pid_gains=sphere_pid_gains,
        sphere_radius=sphere_radius,
        hydroelastic_manipuland=hydroelastic_manipuland,
        directive_files=[scene_directive_path, manipuland_directive_path],
    )
    image_generator_name = f"{prefix}image_generator"
    image_generator_class = IMAGE_GENERATORS[params[image_generator_name]["class"]]
    image_generator = image_generator_class(
        builder=camera_builder,
        scene_graph=camera_scene_graph,
        logger=logger,
        **(params[image_generator_name]["args"] if params[image_generator_name]["args"] is not None else {}),
    )

    images, intrinsics, extrinsics, depths, labels, masks = image_generator.generate_images()
    print(f"Finished generating images{f' for {prefix}' if prefix else ''}.")

    inverse_graphics_name = f"{prefix}inverse_graphics"
    inverse_graphics_class = INVERSE_GRAPHICS[params[inverse_graphics_name]["class"]]
    inverse_graphics = inverse_graphics_class(
        **(params[inverse_graphics_name]["args"] if params[inverse_graphics_name]["args"] is not None else {}),
        images=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth=depths,
        labels=labels,
        masks=masks,
    )
    raw_mesh, raw_mesh_pose = inverse_graphics.run()
    # TODO: Log 'raw_mesh_pose' and 'manipuland_default_pose_transform' as meta-data
    print(f"Finished running inverse graphics{f' for {prefix}' if prefix else ''}.")

    mesh_processor_name = f"{prefix}mesh_processor"
    mesh_processor_class = MESH_PROCESSORS[params[mesh_processor_name]["class"]]
    mesh_processor = mesh_processor_class(
        logger=logger,
        **(params[mesh_processor_name]["args"] if params[mesh_processor_name]["args"] is not None else {}),
    )
    is_primitive, processed_mesh, processed_mesh_pieces, primitive_info = mesh_processor.process_mesh(raw_mesh)
    print(f"Finished mesh processing{f' for {prefix}' if prefix else ''}.")

    # Compute mesh inertia and mass
    physical_porperty_estimator_name = f"{prefix}physical_property_estimator"
    physical_property_estimator_class = PHYSICAL_PROPERTY_ESTIMATOR[params[physical_porperty_estimator_name]["class"]]
    physical_porperty_estimator = physical_property_estimator_class(
        **(
            params[physical_porperty_estimator_name]["args"]
            if params[physical_porperty_estimator_name]["args"] is not None
            else {}
        ),
    )
    mass, inertia = physical_porperty_estimator.estimate_physical_properties(processed_mesh)
    print(f"Finished estimating physical properties{f' for {prefix}' if prefix else ''}.")
    logger.log_manipuland_estimated_physics(manipuland_mass_estimated=mass, manipuland_inertia_estimated=inertia)

    # Save mesh data to create SDF files that can be added to a new simulation environment
    if save_raw_mesh:
        logger.log(raw_mesh=raw_mesh)
    logger.log(processed_mesh=processed_mesh, processed_mesh_piece=processed_mesh_pieces)
    _, processed_mesh_file_path = logger.save_mesh_data(prefix=prefix)
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
            prefix=prefix,
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
            prefix=prefix,
        )

    builder, scene_graph, plant = create_env(
        timestep=timestep,
        env_params=params["env"],
        manipuland_base_link_name=manipuland_base_link_name,
        sphere_starting_position=sphere_starting_position,
        sphere_pid_gains=sphere_pid_gains,
        sphere_radius=sphere_radius,
        hydroelastic_manipuland=hydroelastic_manipuland,
        directive_files=[scene_directive_path],
        directive_strs=[processed_mesh_directive],
        manipuland_pose=RigidTransform(RollPitchYaw(*raw_mesh_pose[:3]), raw_mesh_pose[3:]),
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
    sphere_starting_position: List[float],
    sphere_pid_gains: Dict[str, float],
    sphere_radius: float,
    is_pipeline_comparison: bool,
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
    :param sphere_starting_position: The starting position [x, y, z] of the sphere.
    :param sphere_pid_gains: The PID gains of the inverse dynamics controller. Must contain keys "kp", "ki", and "kd".
    :param sphere_radius: The sphere radius in meters.
    :param is_pipeline_comparison: Whether it is a sim2sim pipeline comparison experiment.
    """
    scene_directive_path = os.path.join(pathlib.Path(__file__).parent.resolve(), SCENE_DIRECTIVE)
    manipuland_directive_path = os.path.join(pathlib.Path(__file__).parent.resolve(), manipuland_directive)

    logger_class = LOGGERS[params["logger"]["class"]]
    logger = logger_class(
        logging_frequency_hz=logging_frequency_hz,
        logging_path=logging_path,
        **(params["logger"]["args"] if params["logger"]["args"] is not None else {}),
    )
    logger.log(experiment_description=params)

    manipuland_default_pose = RigidTransform(RollPitchYaw(*manipuland_default_pose[:3]), manipuland_default_pose[3:])
    if is_pipeline_comparison:
        outer_builder, outer_scene_graph, outer_plant = run_pipeline(
            prefix="outer",
            params=params,
            logger=logger,
            timestep=timestep,
            manipuland_base_link_name=manipuland_base_link_name,
            manipuland_default_pose=manipuland_default_pose,
            save_raw_mesh=save_raw_mesh,
            hydroelastic_manipuland=hydroelastic_manipuland,
            sphere_starting_position=sphere_starting_position,
            sphere_pid_gains=sphere_pid_gains,
            sphere_radius=sphere_radius,
            scene_directive_path=scene_directive_path,
            manipuland_directive_path=manipuland_directive_path,
        )

        inner_builder, inner_scene_graph, inner_plant = run_pipeline(
            prefix="inner",
            params=params,
            logger=logger,
            timestep=timestep,
            manipuland_base_link_name=manipuland_base_link_name,
            manipuland_default_pose=manipuland_default_pose,
            save_raw_mesh=save_raw_mesh,
            hydroelastic_manipuland=hydroelastic_manipuland,
            sphere_starting_position=sphere_starting_position,
            sphere_pid_gains=sphere_pid_gains,
            sphere_radius=sphere_radius,
            scene_directive_path=scene_directive_path,
            manipuland_directive_path=manipuland_directive_path,
        )
    else:
        outer_builder, outer_scene_graph, outer_plant = create_env(
            env_params=params["env"],
            timestep=timestep,
            manipuland_base_link_name=manipuland_base_link_name,
            manipuland_pose=manipuland_default_pose,
            sphere_starting_position=sphere_starting_position,
            sphere_pid_gains=sphere_pid_gains,
            sphere_radius=sphere_radius,
            hydroelastic_manipuland=hydroelastic_manipuland,
            directive_files=[scene_directive_path, manipuland_directive_path],
        )

        inner_builder, inner_scene_graph, inner_plant = run_pipeline(
            params=params,
            logger=logger,
            timestep=timestep,
            manipuland_base_link_name=manipuland_base_link_name,
            manipuland_default_pose=manipuland_default_pose,
            save_raw_mesh=save_raw_mesh,
            hydroelastic_manipuland=hydroelastic_manipuland,
            sphere_starting_position=sphere_starting_position,
            sphere_pid_gains=sphere_pid_gains,
            sphere_radius=sphere_radius,
            scene_directive_path=scene_directive_path,
            manipuland_directive_path=manipuland_directive_path,
        )

    logger.add_plants(outer_plant, inner_plant)
    logger.add_scene_graphs(outer_scene_graph, inner_scene_graph)

    simulator_class = SIMULATORS[params["simulator"]["class"]]
    simulator = simulator_class(
        outer_builder=outer_builder,
        outer_scene_graph=outer_scene_graph,
        inner_builder=inner_builder,
        inner_scene_graph=inner_scene_graph,
        logger=logger,
        is_hydroelastic=params["env"]["contact_model"] != "point",
        **(params["simulator"]["args"] if params["simulator"]["args"] is not None else {}),
    )
    simulator.simulate(sim_duration)
    print("Finished simulating.")

    logger.save_data()
    print("Finished saving data.")
