from typing import List, Tuple, Dict, Union
import os
import pathlib

from pydrake.all import (
    RigidTransform,
    DiagramBuilder,
    SceneGraph,
    MultibodyPlant,
    RollPitchYaw,
)

from .base import run_experiment
from .planar_pushing import create_systems

SCENE_DIRECTIVE = "../../models/random_force/random_force_directive.yaml"


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
    interaction_object_base_link_name: str,
    interaction_object_directive_path: str,
    interaction_object_starting_pose: List[float],
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
    :param interaction_object_base_link_name: The base link name of the interaction
        object.
    :param interaction_object_directive_path: The path to the directive specifying the
        interaction object.
    :param interaction_object_starting_pose: The starting pose
        [roll, pitch, yaw, x, y, z] of the pusher object.
    """

    interaction_object_directive_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), interaction_object_directive_path
    )

    # NOTE: Loading directives with big VTK files can take a long time
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
        directive_files=[*directive_files, interaction_object_directive_path],
        directive_strs=directive_strs,
    )

    plant.SetDefaultFreeBodyPose(
        plant.GetBodyByName(interaction_object_base_link_name),
        RigidTransform(
            RollPitchYaw(*interaction_object_starting_pose[:3]),
            interaction_object_starting_pose[3:],
        ),
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


def run_interaction_pushing(**kwargs):
    return run_experiment(
        create_env_func=create_env, scene_directive=SCENE_DIRECTIVE, **kwargs
    )
