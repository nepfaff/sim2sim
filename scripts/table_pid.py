"""sim2sim entrypoint with the table PID scene."""

#!/bin/python3
import os
import argparse
import pathlib
from typing import List, Tuple

import numpy as np
from pydrake.all import (
    LoadModelDirectives,
    ProcessModelDirectives,
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    BasicVector,
    RigidTransform,
    DiagramBuilder,
    RollPitchYaw,
    PidController,
    SceneGraph,
)

from sim2sim.simulation import TablePIDSimulator
from sim2sim.logging import DynamicLogger
from sim2sim.util import get_parser
from sim2sim.images import SphereImageGenerator

SCENE_DIRECTIVE = "../models/table_pid_scene_directive.yaml"
MANIPULAND_DIRECTIVE = "../models/table_pid_manipuland_directive.yaml"
MANIPULANT_DEFAULT_POSE = RigidTransform(RollPitchYaw(-np.pi / 2.0, 0.0, 0.0), [0.0, 0.0, 0.6])  # X_WManipuland


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

        self._control_output_port = self.DeclareVectorOutputPort("table_angle", BasicVector(2), self.CalcOutput)

    def CalcOutput(self, context, output):
        if context.get_time() < self._no_command_time:
            table_angle = 0.0
        else:
            table_angle = self._angle

        output.SetFromVector([table_angle, 0.0])


def create_env(directives: List[str], args: argparse.Namespace) -> Tuple[DiagramBuilder, SceneGraph]:
    """Creates the table PID simulation environments without building it."""
    # Create plant
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, args.timestep)
    parser = get_parser(plant)
    for directive_path in directives:
        directives = LoadModelDirectives(directive_path)
        ProcessModelDirectives(directives, parser)
    plant.SetDefaultFreeBodyPose(plant.GetBodyByName("ycb_tomato_soup_can_base_link"), MANIPULANT_DEFAULT_POSE)
    plant.Finalize()

    # Table controller
    pid_controller = builder.AddSystem(PidController(kp=np.array([10.0]), ki=np.array([1.0]), kd=np.array([1.0])))
    pid_controller.set_name("pid_controller")

    # Now "wire up" the controller to the plant
    table_instance = plant.GetModelInstanceByName("table")
    builder.Connect(plant.get_state_output_port(table_instance), pid_controller.get_input_port_estimated_state())
    builder.Connect(pid_controller.get_output_port_control(), plant.get_actuation_input_port(table_instance))

    table_angle_source = builder.AddSystem(
        TableAngleSource(args.final_table_angle, no_command_time=args.no_command_time)
    )
    table_angle_source.set_name("table_angle_source")
    builder.Connect(table_angle_source.get_output_port(), pid_controller.get_input_port_desired_state())

    return builder, scene_graph


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--sim_duration",
        type=float,
        default=3.0,
        required=False,
        help="The simulation duration in seconds.",
    )
    argument_parser.add_argument(
        "--timestep",
        type=float,
        default=0.0001,
        required=False,
        help="The timestep to use in seconds.",
    )
    argument_parser.add_argument(
        "--final_table_angle",
        type=float,
        default=np.pi / 4,
        required=False,
        help="The final table angle in radians.",
    )
    argument_parser.add_argument(
        "--no_command_time",
        type=float,
        default=1.0,
        required=False,
        help="The time before starting the table control in seconds.",
    )
    argument_parser.add_argument(
        "--realtime_rate",
        type=float,
        default=1.0,
        required=False,
        help="The simulation realtime rate.",
    )
    argument_parser.add_argument(
        "--html",
        type=str,
        required=False,
        help="Path to save the meshcat html to. The file should end with .html.",
    )
    argument_parser.add_argument(
        "--kProximity", action="store_true", help="Whether to visualize kProximity or kIllustration."
    )
    argument_parser.add_argument("--contact_viz", action="store_true", help="Whether to visualize the contact forces.")
    args = argument_parser.parse_args()
    return args


def main():
    args = parse_args()

    scene_directive = os.path.join(pathlib.Path(__file__).parent.resolve(), SCENE_DIRECTIVE)
    manipuland_directive = os.path.join(pathlib.Path(__file__).parent.resolve(), MANIPULAND_DIRECTIVE)

    # Label 4 is the Tomato Soup Can in this simulation setup
    logger = DynamicLogger(logging_frequency_hz=0.001, logging_path="test_logging_path", label_to_mask=4)

    builder_outer, scene_graph_outer = create_env([scene_directive, manipuland_directive], args)

    # Create a new version of the scene for generating camera data
    builder_camera, scene_graph_camera = create_env([scene_directive, manipuland_directive], args)
    image_generator = SphereImageGenerator(
        builder=builder_camera,
        scene_graph=scene_graph_camera,
        logger=logger,
        simulate_time=args.no_command_time,
        look_at_point=MANIPULANT_DEFAULT_POSE.translation(),
        z_distances=[0.02, 0.2, 0.3],
        radii=[0.4, 0.3, 0.3],
        num_poses=[30, 25, 15],
    )

    images, intrinsics, extrinsics, depths, labels, masks = image_generator.generate_images()
    print("Finished generating images.")

    # TODO: Replace the following builder and scene graph with the ones generated from InverseGraphics
    builder_inner, scene_graph_inner = create_env([scene_directive, manipuland_directive], args)

    simulator = TablePIDSimulator(builder_outer, scene_graph_outer, builder_inner, scene_graph_inner, logger)
    simulator.simulate(args.sim_duration)

    print("Saving data.")
    logger.save_data()


if __name__ == "__main__":
    main()
