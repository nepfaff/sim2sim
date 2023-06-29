from typing import List, Tuple

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
    SpatialInertia,
    UnitInertia,
    PrismaticJoint,
    LeafSystem,
)
from manipulation.scenarios import AddShape

from sim2sim.util import get_parser, ExternalForceSystem
from .base import run_experiment

SCENE_DIRECTIVE = "../../models/random_force/random_force_directive.yaml"


def add_point_finger(
    plant: MultibodyPlant,
    radius: float = 0.01,
    position: List[float] = [0.0, 0.0, -1.0],
) -> None:
    finger = AddShape(plant, Sphere(radius), "point_finger", color=[0.9, 0.5, 0.5, 1.0])
    _ = plant.AddRigidBody(
        "false_body1", finger, SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0))
    )
    finger_x = plant.AddJoint(
        PrismaticJoint(
            "finger_x",
            plant.world_frame(),
            plant.GetFrameByName("false_body1"),
            [1, 0, 0],
            -1.0,
            1.0,
        )
    )
    finger_x.set_default_translation(position[0])
    plant.AddJointActuator("finger_x", finger_x)
    _ = plant.AddRigidBody(
        "false_body2", finger, SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0))
    )
    finger_y = plant.AddJoint(
        PrismaticJoint(
            "finger_y",
            plant.GetFrameByName("false_body1"),
            plant.GetFrameByName("false_body2"),
            [0, 1, 0],
            -1.0,
            1.0,
        )
    )
    finger_y.set_default_translation(position[1])
    plant.AddJointActuator("finger_y", finger_y)
    finger_z = plant.AddJoint(
        PrismaticJoint(
            "finger_z",
            plant.GetFrameByName("false_body2"),
            plant.GetFrameByName("point_finger"),
            [0, 0, 1],
            -1.0,
            1.0,
        )
    )
    finger_z.set_default_translation(position[2])
    plant.AddJointActuator("finger_z", finger_z)


class PointFingerForceControl(LeafSystem):
    def __init__(self, plant: MultibodyPlant, finger_mass: float = 1.0):
        LeafSystem.__init__(self)
        self._plant = plant
        self._finger_mass = finger_mass

        self.DeclareVectorInputPort("desired_contact_force", 3)
        self.DeclareVectorOutputPort("finger_actuation", 3, self.CalcOutput)

    def CalcOutput(self, context, output):
        g = self._plant.gravity_field().gravity_vector()

        desired_force = self.get_input_port(0).Eval(context)
        output.SetFromVector(-self._finger_mass * g - desired_force)


def create_env(
    env_params: dict,
    timestep: float,
    manipuland_base_link_name: str,
    manipuland_pose: RigidTransform,
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
    **kwargs,
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """
    Creates the random force simulation environments without building it.

    :param env_params: The dict containing environment specific parameters.
    :param timestep: The timestep to use in seconds.
    :param manipuland_base_link_name: The base link name of the outer manipuland.
    :param manipuland_pose: The default pose of the outer manipuland of form
        [roll, pitch, yaw, x, y, z].
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

    add_point_finger(plant)
    point_finger_controller = builder.AddSystem(PointFingerForceControl(plant))

    plant.SetDefaultFreeBodyPose(
        plant.GetBodyByName(manipuland_base_link_name), manipuland_pose
    )
    plant.Finalize()

    builder.Connect(
        point_finger_controller.get_output_port(), plant.get_actuation_input_port()
    )
    builder.ExportInput(
        point_finger_controller.get_input_port(), "desired_contact_force"
    )

    external_force_system = builder.AddSystem(
        ExternalForceSystem(plant.GetBodyByName(manipuland_base_link_name).index())
    )
    external_force_system.set_name("external_force_system")
    builder.Connect(
        external_force_system.get_output_port(),
        plant.get_applied_spatial_force_input_port(),
    )

    return builder, scene_graph, plant


def run_random_force(**kwargs):
    return run_experiment(
        create_env_func=create_env, scene_directive=SCENE_DIRECTIVE, **kwargs
    )
