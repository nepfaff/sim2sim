from typing import List, Tuple

import numpy as np
from pydrake.all import (
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    ProcessModelDirectives,
    AddMultibodyPlant,
    RigidTransform,
    DiagramBuilder,
    SceneGraph,
    MultibodyPlant,
    Demultiplexer,
    MultibodyPlantConfig,
)

from sim2sim.util import (
    get_parser,
    add_iiwa_system,
    add_cameras,
    add_wsg_system,
    IIWAJointTrajectorySource,
    WSGCommandSource,
    IIWAControlModeSource,
)
from .base import run_experiment

SCENE_DIRECTIVE = "../../models/iiwa_manip/iiwa_manip_scene_directive.yaml"
# iiwa joint angles in radians
IIWA_Q_NOMINAL = np.array([1.5, -0.4, 0.0, -1.75, 0.0, 1.5, 0.0])


def create_env(
    timestep: float,
    manipuland_pose: RigidTransform,
    manipuland_base_link_name: str,
    weld_manipuland: bool = False,
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
    **kwargs,
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """
    Creates the iiwa rearrangement simulation environments without building it.

    :param env_params: The dict containing environment specific parameters.
    :param timestep: The timestep to use in seconds.
    :param manipuland_base_link_name: The base link name of the outer manipuland.
    :param weld_manipuland: Whether to weld the manipuland to the world.
    :param manipuland_pose: The default pose of the outer manipuland of form
        [roll, pitch, yaw, x, y, z].
    """

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

    manipuland_body = plant.GetBodyByName(manipuland_base_link_name)
    plant.SetDefaultFreeBodyPose(manipuland_body, manipuland_pose)
    if weld_manipuland:
        plant.WeldFrames(
            plant.world_frame(), manipuland_body.body_frame(), manipuland_pose
        )

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
        builder=builder,
        plant=plant,
        iiwa_instance_idx=plant.GetModelInstanceByName("iiwa"),
        iiwa_time_step=timestep,
    )
    iiwa_control_mode_source = builder.AddSystem(IIWAControlModeSource())
    iiwa_control_mode_source.set_name("iiwa_control_mode_source")
    builder.Connect(
        iiwa_control_mode_source.GetOutputPort("iiwa_control_mode"),
        iiwa_control_mode_input,
    )

    # Add wsg controller
    (
        wsg_position_input,
        wsg_force_limit_input,
        wsg_state_measured_output,
        wsg_force_measured_output,
    ) = add_wsg_system(
        builder=builder,
        plant=plant,
        wsg_instance_idx=plant.GetModelInstanceByName("wsg"),
        wsg_time_step=timestep,
    )

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
    builder.Connect(
        iiwa_joint_trajectory_source.get_output_port(), demux.get_input_port()
    )
    builder.Connect(demux.get_output_port(0), iiwa_position_input)

    # Add wsg position source
    wsg_command_source = builder.AddSystem(WSGCommandSource(initial_pos=0.1))
    wsg_command_source.set_name("wsg_command_source")
    builder.Connect(wsg_command_source.get_output_port(), wsg_position_input)

    return builder, scene_graph, plant


def run_iiwa_manip(**kwargs):
    return run_experiment(
        create_env_func=create_env, scene_directive=SCENE_DIRECTIVE, **kwargs
    )
