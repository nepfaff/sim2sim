from typing import List, Tuple

import numpy as np

from pydrake.all import (
    AddMultibodyPlant,
    BasicVector,
    DiagramBuilder,
    LeafSystem,
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    MultibodyPlant,
    MultibodyPlantConfig,
    PidController,
    ProcessModelDirectives,
    RigidTransform,
    SceneGraph,
)

from sim2sim.util import get_parser

from .base import run_experiment

SCENE_DIRECTIVE = "../../models/table_pid/table_pid_scene_directive.yaml"


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

        self._control_output_port = self.DeclareVectorOutputPort(
            "table_angle", BasicVector(2), self.CalcOutput
        )

    def CalcOutput(self, context, output):
        if context.get_time() < self._no_command_time:
            table_angle = 0.0
        else:
            table_angle = self._angle

        output.SetFromVector([table_angle, 0.0])


def create_env(
    env_params: dict,
    timestep: float,
    final_table_angle: float,
    no_command_time: float,
    manipuland_base_link_names: List[str],
    manipuland_poses: List[RigidTransform],
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
    **kwargs,
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """
    Creates the table PID simulation environments without building it.

    :param env_params: The dict containing environment specific parameters.
    :param timestep: The timestep to use in seconds.
    :param manipuland_base_link_names: The base link names of the outer manipulands.
    :param manipuland_poses: The default poses of the outer manipulands of form
        [roll, pitch, yaw, x, y, z].
    """
    # Create plant
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

    for link_name, pose in zip(manipuland_base_link_names, manipuland_poses):
        plant.SetDefaultFreeBodyPose(plant.GetBodyByName(link_name), pose)
    plant.Finalize()

    # Table controller
    pid_controller = builder.AddSystem(
        PidController(kp=np.array([10.0]), ki=np.array([1.0]), kd=np.array([1.0]))
    )
    pid_controller.set_name("pid_controller")

    # Now "wire up" the controller to the plant
    table_instance = plant.GetModelInstanceByName("table")
    builder.Connect(
        plant.get_state_output_port(table_instance),
        pid_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        pid_controller.get_output_port_control(),
        plant.get_actuation_input_port(table_instance),
    )

    table_angle_source = builder.AddSystem(
        TableAngleSource(final_table_angle, no_command_time=no_command_time)
    )
    table_angle_source.set_name("table_angle_source")
    builder.Connect(
        table_angle_source.get_output_port(),
        pid_controller.get_input_port_desired_state(),
    )

    return builder, scene_graph, plant


def run_table_pid(**kwargs):
    return run_experiment(
        create_env_func=create_env, scene_directive=SCENE_DIRECTIVE, **kwargs
    )
