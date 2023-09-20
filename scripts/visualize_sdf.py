"""Script for visualizing Drake SDF files in meshcat."""

import argparse
import time

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    Simulator,
    StartMeshcat,
)

from sim2sim.util import get_parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sdf_path",
        required=True,
        type=str,
        help="The path to the sdf file to visualize.",
    )
    parser.add_argument(
        "--kProximity",
        action="store_true",
        help="Whether to visualize kProximity or kIllustration.",
    )
    args = parser.parse_args()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = get_parser(plant)

    parser.AddModelFromFile(args.sdf_path)

    meshcat = StartMeshcat()
    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.role = Role.kProximity if args.kProximity else Role.kIllustration
    _ = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph.get_query_output_port(), meshcat, meshcat_params
    )

    plant.Finalize()
    diagram = builder.Build()

    # Need to simulate for visualization to work
    simulator = Simulator(diagram)
    simulator.AdvanceTo(0.0)

    # Sleep to give user enough time to click on meshcat link
    time.sleep(10.0)


if __name__ == "__main__":
    main()
