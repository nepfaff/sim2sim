from typing import List, Tuple, Dict, Union

from pydrake.all import (
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    ProcessModelDirectives,
    RigidTransform,
    DiagramBuilder,
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

from sim2sim.util import (
    get_parser,
    StateSource,
    copy_object_proximity_properties,
    add_shape,
)
from .base import run_experiment

SCENE_DIRECTIVE = "../../models/random_force/random_force_directive.yaml"


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


def create_systems(
    env_params: dict,
    timestep: float,
    manipuland_base_link_name: str,
    manipuland_pose: RigidTransform,
    hydroelastic_manipuland: bool,
    pusher_geometry_type: str,
    pusher_geometry_starting_position: List[float],
    pusher_geometry_pid_gains: Dict[str, float],
    pusher_geometry_dimensions: Union[float, List[float]],
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
) -> Tuple[
    DiagramBuilder, SceneGraph, MultibodyPlant, InverseDynamicsController, StateSource
]:
    """
    Creates the systems for the planar pushing environment without finalizing the plant.
    The systems still need to be connected.
    """
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
        StateSource(pusher_geometry_starting_position)
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

    return (
        builder,
        scene_graph,
        plant,
        pusher_geometry_inverse_dynamics_controller,
        pusher_geometry_state_source,
    )


def create_env(
    env_params: dict,
    timestep: float,
    manipuland_base_link_name: str,
    manipuland_pose: RigidTransform,
    hydroelastic_manipuland: bool,
    pusher_geometry_type: str,
    pusher_geometry_starting_position: List[float],
    pusher_geometry_pid_gains: Dict[str, float],
    pusher_geometry_dimensions: Union[float, List[float]],
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """
    Creates the planar pushing simulation environment without building it.

    :param env_params: The dict containing environment specific parameters.
    :param timestep: The timestep to use in seconds.
    :param manipuland_base_link_name: The base link name of the outer manipuland.
    :param manipuland_pose: The default pose of the outer manipuland.
    :param hydroelastic_manipuland: Whether to use hydroelastic or point contact for the
        inner manipuland.
    :param pusher_geometry_type: The pusher geometry type. "sphere" or "box".
    :param pusher_geometry_starting_position: The starting position [x, y, z] of the
        pusher_geometry.
    :param pusher_geometry_pid_gains: The PID gains of the inverse dynamics controller.
        Must contain keys "kp", "ki", and "kd".
    :param pusher_geometry_dimensions: The dimensions for the pusher geometry. Radius
        for a pusher_geometry and [W,D,H] for a pusher geometry.
    """

    (
        builder,
        scene_graph,
        plant,
        pusher_geometry_inverse_dynamics_controller,
        pusher_geometry_state_source,
    ) = create_systems(
        env_params=env_params,
        timestep=timestep,
        manipuland_base_link_name=manipuland_base_link_name,
        manipuland_pose=manipuland_pose,
        hydroelastic_manipuland=hydroelastic_manipuland,
        pusher_geometry_type=pusher_geometry_type,
        pusher_geometry_starting_position=pusher_geometry_starting_position,
        pusher_geometry_pid_gains=pusher_geometry_pid_gains,
        pusher_geometry_dimensions=pusher_geometry_dimensions,
        directive_files=directive_files,
        directive_strs=directive_strs,
    )

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


def run_planar_pushing(**kwargs):
    return run_experiment(
        create_env_func=create_env, scene_directive=SCENE_DIRECTIVE, **kwargs
    )
