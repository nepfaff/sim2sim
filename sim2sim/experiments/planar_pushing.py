import os
import pathlib
from typing import List, Tuple, Dict, Union

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
    Box,
    SpatialInertia,
    UnitInertia,
    PrismaticJoint,
    InverseDynamicsController,
    AddCompliantHydroelasticProperties,
    ProximityProperties,
    RoleAssign,
)

from sim2sim.simulation import (
    BasicSimulator,
    PlanarPushingSimulator,
    EquationErrorPlanarPushingSimulator,
)
from sim2sim.logging import PlanarPushingLogger
from sim2sim.util import (
    get_parser,
    create_processed_mesh_directive_str,
    create_processed_mesh_primitive_directive_str,
    create_directive_str_for_sdf_path,
    SphereStateSource,
    copy_object_proximity_properties,
    add_shape,
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
    IdentitySDFMeshProcessor,
)
from sim2sim.physical_property_estimator import (
    WaterDensityPhysicalPropertyEstimator,
    GTPhysicalPropertyEstimator,
)

SCENE_DIRECTIVE = "../../models/random_force/random_force_directive.yaml"

# TODO: Add type info using base classes
LOGGERS = {
    "PlanarPushingLogger": PlanarPushingLogger,
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
    "IdentitySDFMeshProcessor": IdentitySDFMeshProcessor,
}
PHYSICAL_PROPERTY_ESTIMATOR = {
    "WaterDensityPhysicalPropertyEstimator": WaterDensityPhysicalPropertyEstimator,
    "GTPhysicalPropertyEstimator": GTPhysicalPropertyEstimator,
}
SIMULATORS = {
    "BasicSimulator": BasicSimulator,
    "PlanarPushingSimulator": PlanarPushingSimulator,
    "EquationErrorPlanarPushingSimulator": EquationErrorPlanarPushingSimulator,
}


def add_pusher_geometry(
    plant: MultibodyPlant,
    type: str,
    dimensions: Union[float, List[float]],
    position: List[float] = [0.0, 0.0, 0.0],
) -> None:
    if type.lower() == "sphere":
        assert isinstance(dimensions, float)
        pusher_geometry = Sphere(dimensions)
    elif type.lower() == "box":
        assert isinstance(dimensions, List) and len(dimensions) == 3
        pusher_geometry = Box(dimensions[0], dimensions[1], dimensions[2])
    else:
        print(f"Unknown pusher geometry: {type}")
        exit()
    pusher_shape = add_shape(
        plant,
        pusher_geometry,
        "pusher_geometry",
        color=[0.9, 0.5, 0.5, 1.0],
    )
    _ = plant.AddRigidBody(
        "false_body1", pusher_shape, SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0))
    )
    pusher_geometry_x = plant.AddJoint(
        PrismaticJoint(
            "pusher_geometry_x",
            plant.world_frame(),
            plant.GetFrameByName("false_body1"),
            [1, 0, 0],
            -10.0,
            10.0,
        )
    )
    pusher_geometry_x.set_default_translation(position[0])
    plant.AddJointActuator("pusher_geometry_x", pusher_geometry_x)
    _ = plant.AddRigidBody(
        "false_body2", pusher_shape, SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0))
    )
    pusher_geometry_y = plant.AddJoint(
        PrismaticJoint(
            "pusher_geometry_y",
            plant.GetFrameByName("false_body1"),
            plant.GetFrameByName("false_body2"),
            [0, 1, 0],
            -10.0,
            10.0,
        )
    )
    pusher_geometry_y.set_default_translation(position[1])
    plant.AddJointActuator("pusher_geometry_y", pusher_geometry_y)
    pusher_geometry_z = plant.AddJoint(
        PrismaticJoint(
            "pusher_geometry_z",
            plant.GetFrameByName("false_body2"),
            plant.GetFrameByName("pusher_geometry"),
            [0, 0, 1],
            -10.0,
            10.0,
        )
    )
    pusher_geometry_z.set_default_translation(position[2])
    plant.AddJointActuator("pusher_geometry_z", pusher_geometry_z)


def create_env(
    env_params: dict,
    timestep: float,
    manipuland_base_link_name: str,
    manipuland_pose: RigidTransform,
    pusher_geometry_starting_position: List[float],
    pusher_geometry_pid_gains: Dict[str, float],
    hydroelastic_manipuland: bool,
    pusher_geometry_type: str,
    pusher_geometry_dimensions: Union[float, List[float]],
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """Creates the planar pushing simulation environment without building it."""

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

    add_pusher_geometry(
        plant,
        type=pusher_geometry_type,
        dimensions=pusher_geometry_dimensions,
        position=pusher_geometry_starting_position,
    )
    if hydroelastic_manipuland:
        print(
            "Enabling hydroelastic for pusher_geometry. This might take a while. Increase the "
            + "resolution hint to make this faster."
        )
        # Make pusher_geometry complient hydroelastic
        pusher_geometry = plant.GetBodyByName("pusher_geometry")
        geometry_ids = plant.GetCollisionGeometriesForBody(pusher_geometry)
        inspector = scene_graph.model_inspector()
        for geometry_id in geometry_ids:
            new_proximity_properties = ProximityProperties()
            # NOTE: Setting hydroelastic properties becomes slow as the resolution hint decreases
            AddCompliantHydroelasticProperties(
                resolution_hint=0.1,
                hydroelastic_modulus=1e8,
                properties=new_proximity_properties,
            )
            const_proximity_properties = inspector.GetProximityProperties(geometry_id)
            copy_object_proximity_properties(
                const_proximity_properties, new_proximity_properties
            )
            scene_graph.AssignRole(
                plant.get_source_id(),
                geometry_id,
                new_proximity_properties,
                RoleAssign.kReplace,
            )

    # pusher_geometry state source
    pusher_geometry_state_source = builder.AddSystem(
        SphereStateSource(pusher_geometry_starting_position)
    )
    pusher_geometry_state_source.set_name("pusher_geometry_state_source")

    # pusher_geometry controller
    pusher_geometry_controller_plant = MultibodyPlant(time_step=timestep)
    pusher_geometry_controller_plant.set_name("pusher_geometry_controller_plant")
    add_pusher_geometry(
        pusher_geometry_controller_plant,
        type=pusher_geometry_type,
        dimensions=pusher_geometry_dimensions,
    )
    pusher_geometry_controller_plant.Finalize()
    pusher_geometry_inverse_dynamics_controller = builder.AddSystem(
        InverseDynamicsController(
            pusher_geometry_controller_plant,
            kp=[pusher_geometry_pid_gains["kp"]] * 3,
            ki=[pusher_geometry_pid_gains["ki"]] * 3,
            kd=[pusher_geometry_pid_gains["kd"]] * 3,
            has_reference_acceleration=False,
        )
    )
    pusher_geometry_inverse_dynamics_controller.set_name(
        "pusher_geometry_inverse_dynamics_controller"
    )

    plant.SetDefaultFreeBodyPose(
        plant.GetBodyByName(manipuland_base_link_name), manipuland_pose
    )
    plant.Finalize()

    # Connect pusher_geometry state source and controller to plant
    pusher_geometry_instance = plant.GetModelInstanceByName("pusher_geometry")
    builder.Connect(
        plant.get_state_output_port(pusher_geometry_instance),
        pusher_geometry_inverse_dynamics_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        pusher_geometry_inverse_dynamics_controller.get_output_port_control(),
        plant.get_actuation_input_port(pusher_geometry_instance),
    )
    builder.Connect(
        pusher_geometry_state_source.get_output_port(),
        pusher_geometry_inverse_dynamics_controller.get_input_port_desired_state(),
    )

    return builder, scene_graph, plant


def run_pipeline(
    params: dict,
    logger: PlanarPushingLogger,
    timestep: float,
    manipuland_base_link_name: str,
    manipuland_default_pose: RigidTransform,
    save_raw_mesh: bool,
    hydroelastic_manipuland: bool,
    pusher_geometry_starting_position: List[float],
    pusher_geometry_pid_gains: Dict[str, float],
    pusher_geometry_type: str,
    pusher_geometry_dimensions: Union[float, List[float]],
    scene_directive_path: str,
    manipuland_directive_path: str,
    prefix: str = "",
):
    """
    Runs the sim2sim pipeline of camera data generation, mesh generation, mesh
    processing, and physical property estimation.

    :param prefix: The prefix of the pipeline components in `params`.
    """
    prefix = prefix + "_" if prefix else ""

    # Create a new version of the scene for generating camera data
    camera_builder, camera_scene_graph, _ = create_env(
        env_params=params[f"{prefix}env"],
        timestep=timestep,
        manipuland_base_link_name=manipuland_base_link_name,
        manipuland_pose=manipuland_default_pose,
        pusher_geometry_starting_position=pusher_geometry_starting_position,
        pusher_geometry_pid_gains=pusher_geometry_pid_gains,
        pusher_geometry_type=pusher_geometry_type,
        pusher_geometry_dimensions=pusher_geometry_dimensions,
        hydroelastic_manipuland=hydroelastic_manipuland,
        directive_files=[scene_directive_path, manipuland_directive_path],
    )
    image_generator_name = f"{prefix}image_generator"
    image_generator_class = IMAGE_GENERATORS[params[image_generator_name]["class"]]
    image_generator = image_generator_class(
        builder=camera_builder,
        scene_graph=camera_scene_graph,
        logger=logger,
        **(
            params[image_generator_name]["args"]
            if params[image_generator_name]["args"] is not None
            else {}
        ),
    )

    (
        images,
        intrinsics,
        extrinsics,
        depths,
        labels,
        masks,
    ) = image_generator.generate_images()
    print(f"Finished generating images{f' for {prefix}' if prefix else ''}.")

    inverse_graphics_name = f"{prefix}inverse_graphics"
    inverse_graphics_class = INVERSE_GRAPHICS[params[inverse_graphics_name]["class"]]
    inverse_graphics = inverse_graphics_class(
        **(
            params[inverse_graphics_name]["args"]
            if params[inverse_graphics_name]["args"] is not None
            else {}
        ),
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
        **(
            params[mesh_processor_name]["args"]
            if params[mesh_processor_name]["args"] is not None
            else {}
        ),
    )
    (
        is_primitive,
        processed_mesh,
        processed_mesh_pieces,
        primitive_info,
        mesh_sdf_path,
    ) = mesh_processor.process_mesh(raw_mesh)
    print(f"Finished mesh processing{f' for {prefix}' if prefix else ''}.")

    # Compute mesh inertia and mass
    physical_porperty_estimator_name = f"{prefix}physical_property_estimator"
    physical_property_estimator_class = PHYSICAL_PROPERTY_ESTIMATOR[
        params[physical_porperty_estimator_name]["class"]
    ]
    physical_porperty_estimator = physical_property_estimator_class(
        **(
            params[physical_porperty_estimator_name]["args"]
            if params[physical_porperty_estimator_name]["args"] is not None
            else {}
        ),
    )
    (
        mass,
        inertia,
        center_of_mass,
    ) = physical_porperty_estimator.estimate_physical_properties(processed_mesh)
    print(
        f"Finished estimating physical properties{f' for {prefix}' if prefix else ''}."
    )
    logger.log_manipuland_estimated_physics(
        manipuland_mass_estimated=mass,
        manipuland_inertia_estimated=inertia,
        manipuland_com_estimated=center_of_mass,
    )

    # Save mesh data to create SDF files that can be added to a new simulation environment
    if save_raw_mesh:
        logger.log(raw_mesh=raw_mesh)
    logger.log(
        processed_mesh=processed_mesh, processed_mesh_piece=processed_mesh_pieces
    )
    _, processed_mesh_file_path = logger.save_mesh_data(prefix=prefix)
    processed_mesh_file_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "../..", processed_mesh_file_path
    )

    # Create a directive for processed_mesh manipuland
    if mesh_sdf_path is not None:
        processed_mesh_directive = create_directive_str_for_sdf_path(
            mesh_sdf_path, params[f"{prefix}env"]["obj_name"]
        )
    elif is_primitive:
        processed_mesh_directive = create_processed_mesh_primitive_directive_str(
            primitive_info,
            mass,
            inertia,
            center_of_mass,
            logger._mesh_dir_path,
            params[f"{prefix}env"]["obj_name"],
            manipuland_base_link_name,
            hydroelastic=hydroelastic_manipuland,
            prefix=prefix,
        )
    else:
        processed_mesh_directive = create_processed_mesh_directive_str(
            mass,
            inertia,
            center_of_mass,
            processed_mesh_file_path,
            logger._mesh_dir_path,
            params[f"{prefix}env"]["obj_name"],
            manipuland_base_link_name,
            hydroelastic=hydroelastic_manipuland,
            prefix=prefix,
        )

    builder, scene_graph, plant = create_env(
        timestep=timestep,
        env_params=params[f"{prefix}env"],
        manipuland_base_link_name=manipuland_base_link_name,
        pusher_geometry_starting_position=pusher_geometry_starting_position,
        pusher_geometry_pid_gains=pusher_geometry_pid_gains,
        pusher_geometry_type=pusher_geometry_type,
        pusher_geometry_dimensions=pusher_geometry_dimensions,
        hydroelastic_manipuland=hydroelastic_manipuland,
        directive_files=[scene_directive_path],
        directive_strs=[processed_mesh_directive],
        manipuland_pose=RigidTransform(
            RollPitchYaw(*raw_mesh_pose[:3]), raw_mesh_pose[3:]
        ),
    )

    return builder, scene_graph, plant


def run_planar_pushing(
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
    pusher_geometry_starting_position: List[float],
    pusher_geometry_pid_gains: Dict[str, float],
    pusher_geometry_type: str,
    pusher_geometry_dimensions: Union[float, List[float]],
    is_pipeline_comparison: bool,
):
    """
    Experiment entrypoint for the planar pushing scene.

    :param logging_path: The path to log the data to.
    :param params: The experiment yaml file dict.
    :param sim_duration: The simulation duration in seconds.
    :param timestep: The timestep to use in seconds.
    :param logging_frequency_hz: The dynamics logging frequency.
    :param manipuland_directive: The file path of the outer manipuland directive. The
     path should be relative to this script.
    :param manipuland_base_link_name: The base link name of the outer manipuland.
    :param manipuland_default_pose: The default pose of the outer manipuland of form
        [roll, pitch, yaw, x, y, z].
    :param save_raw_mesh: Whether to save the raw mesh from inverse graphics.
    :param hydroelastic_manipuland: Whether to use hydroelastic or point contact for the
        inner manipuland.
    :param pusher_geometry_starting_position: The starting position [x, y, z] of the pusher_geometry.
    :param pusher_geometry_pid_gains: The PID gains of the inverse dynamics controller. Must
        contain keys "kp", "ki", and "kd".
    :param pusher_geometry_type: The pusher geometry type. "sphere" or "box".
    :param pusher_geometry_dimensions: The dimensions for the pusher geometry. Radius
        for a pusher_geometry and [W,D,H] for a pusher geometry.
    :param is_pipeline_comparison: Whether it is a sim2sim pipeline comparison experiment.
    """
    scene_directive_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), SCENE_DIRECTIVE
    )
    manipuland_directive_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), manipuland_directive
    )

    logger_class = LOGGERS[params["logger"]["class"]]
    logger = logger_class(
        logging_frequency_hz=logging_frequency_hz,
        logging_path=logging_path,
        **(params["logger"]["args"] if params["logger"]["args"] is not None else {}),
    )
    logger.log(experiment_description=params)

    manipuland_default_pose = RigidTransform(
        RollPitchYaw(*manipuland_default_pose[:3]), manipuland_default_pose[3:]
    )
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
            pusher_geometry_starting_position=pusher_geometry_starting_position,
            pusher_geometry_pid_gains=pusher_geometry_pid_gains,
            pusher_geometry_type=pusher_geometry_type,
            pusher_geometry_dimensions=pusher_geometry_dimensions,
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
            pusher_geometry_starting_position=pusher_geometry_starting_position,
            pusher_geometry_pid_gains=pusher_geometry_pid_gains,
            pusher_geometry_type=pusher_geometry_type,
            pusher_geometry_dimensions=pusher_geometry_dimensions,
            scene_directive_path=scene_directive_path,
            manipuland_directive_path=manipuland_directive_path,
        )
    else:
        outer_builder, outer_scene_graph, outer_plant = create_env(
            env_params=params["env"],
            timestep=timestep,
            manipuland_base_link_name=manipuland_base_link_name,
            manipuland_pose=manipuland_default_pose,
            pusher_geometry_starting_position=pusher_geometry_starting_position,
            pusher_geometry_pid_gains=pusher_geometry_pid_gains,
            pusher_geometry_type=pusher_geometry_type,
            pusher_geometry_dimensions=pusher_geometry_dimensions,
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
            pusher_geometry_starting_position=pusher_geometry_starting_position,
            pusher_geometry_pid_gains=pusher_geometry_pid_gains,
            pusher_geometry_type=pusher_geometry_type,
            pusher_geometry_dimensions=pusher_geometry_dimensions,
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
        is_hydroelastic=params[f"{'inner_' if is_pipeline_comparison else ''}env"][
            "contact_model"
        ]
        != "point",  # Visualize outer contact forces if inner/outer use different contact engines
        **(
            params["simulator"]["args"]
            if params["simulator"]["args"] is not None
            else {}
        ),
    )
    simulator.simulate(sim_duration)
    print("Finished simulating.")

    logger.save_data()
    print("Finished saving data.")