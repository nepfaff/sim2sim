"""sim2sim entrypoint with the table PID scene."""

#!/bin/python3
import argparse
import os
from typing import Tuple

import numpy as np
from pydrake.all import (
    Parser,
    LoadModelDirectivesFromString,
    ProcessModelDirectives,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    BasicVector,
    RigidTransform,
    DiagramBuilder,
    RollPitchYaw,
    SceneGraph,
    PidController,
)
from manipulation.scenarios import AddPackagePaths

from sim2sim.simulation import TablePIDSimulator
from sim2sim.logging import DynamicLogger

MESH_MANIPULANT_DEFAULT_POSE = RigidTransform(RollPitchYaw(0.0, 0.0, np.pi / 2), [0.0, 0.0, 0.7])  # X_WMesh


def create_plant(model_directives: str, time_step: float) -> Tuple[DiagramBuilder, MultibodyPlant, SceneGraph, Parser]:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    parser = Parser(plant)
    parser.package_map().Add("sim2sim", os.path.abspath(""))
    AddPackagePaths(parser)
    directives = LoadModelDirectivesFromString(model_directives)
    ProcessModelDirectives(directives, parser)
    plant.SetDefaultFreeBodyPose(plant.GetBodyByName("ycb_tomato_soup_can_base_link"), MESH_MANIPULANT_DEFAULT_POSE)
    plant.Finalize()
    return builder, plant, scene_graph, parser


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim_duration",
        type=float,
        default=3.0,
        required=False,
        help="The simulation duration in seconds.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.0001,
        required=False,
        help="The timestep to use.",
    )
    parser.add_argument(
        "--final_table_angle",
        type=float,
        default=np.pi / 4,
        required=False,
        help="The final table angle.",
    )
    parser.add_argument(
        "--no_command_time",
        type=float,
        default=2.0,
        required=False,
        help="The time before starting the table control.",
    )
    parser.add_argument(
        "--realtime_rate",
        type=float,
        default=1.0,
        required=False,
        help="The simulation realtime rate.",
    )
    parser.add_argument(
        "--html",
        type=str,
        required=False,
        help="Path to save the meshcat html to. The file should end with .html.",
    )
    parser.add_argument("--kProximity", action="store_true", help="Whether to visualize kProximity or kIllustration.")
    parser.add_argument("--contact_viz", action="store_true", help="Whether to visualize the contact forces.")
    args = parser.parse_args()

    # TODO: Split manipulant into separate directive file. This way we can build the outer and inner simulation in the
    # same way and then simply add different manipulands to them.
    model_directives = """
    directives:
    - add_directives:
        file: package://sim2sim/models/table_pid_directive.yaml
    """

    builder, plant, scene_graph, parser = create_plant(model_directives, time_step=args.timestep)

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

    ## NOTE: Start temp
    ## NOTE: This represents the inner simulation environment
    builder2, plant2, scene_graph2, parser2 = create_plant(model_directives, time_step=args.timestep)
    pid_controller2 = builder2.AddSystem(PidController(kp=np.array([10.0]), ki=np.array([1.0]), kd=np.array([1.0])))
    pid_controller2.set_name("pid_controller")
    table_instance2 = plant2.GetModelInstanceByName("table")
    builder2.Connect(plant2.get_state_output_port(table_instance2), pid_controller2.get_input_port_estimated_state())
    builder2.Connect(pid_controller2.get_output_port_control(), plant2.get_actuation_input_port(table_instance2))
    table_angle_source2 = builder2.AddSystem(
        TableAngleSource(args.final_table_angle, no_command_time=args.no_command_time)
    )
    table_angle_source2.set_name("table_angle_source")
    builder2.Connect(table_angle_source2.get_output_port(), pid_controller2.get_input_port_desired_state())
    ## NOTE: End temp

    logger = DynamicLogger(logging_frequency_hz=0.001, logging_path="test_logging_path")
    simulator = TablePIDSimulator(builder, scene_graph, builder2, scene_graph2, logger)
    simulator.simulate(args.sim_duration)


if __name__ == "__main__":
    main()
