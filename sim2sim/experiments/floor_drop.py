from typing import List, Tuple

from pydrake.all import (
    AddMultibodyPlant,
    DiagramBuilder,
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    MultibodyPlant,
    MultibodyPlantConfig,
    ProcessModelDirectives,
    RigidTransform,
    SceneGraph,
)

from sim2sim.util import get_parser

from .base import run_experiment

SCENE_DIRECTIVE = "../../models/floor_drop/floor_drop_directive.yaml"


def create_env(
    env_params: dict,
    timestep: float,
    manipuland_base_link_names: List[str],
    manipuland_poses: List[RigidTransform],
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
    **kwargs,
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """
    Creates the floor drop simulation environments without building it.

    :param env_params: The dict containing environment specific parameters.
    :param timestep: The timestep to use in seconds.
    :param manipuland_base_link_names: The base link names of the outer manipulands.
    :param manipuland_poses: The default poses of the outer manipulands.
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

    for link_name, pose in zip(manipuland_base_link_names, manipuland_poses):
        plant.SetDefaultFreeBodyPose(plant.GetBodyByName(link_name), pose)
    plant.Finalize()

    return builder, scene_graph, plant


def run_floor_drop(**kwargs):
    return run_experiment(
        create_env_func=create_env, scene_directive=SCENE_DIRECTIVE, **kwargs
    )
