import os
import shutil
import pathlib
from typing import List, Tuple

import numpy as np
from pydrake.all import (
    LoadModelDirectives,
    LoadModelDirectivesFromString,
    ProcessModelDirectives,
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    BasicVector,
    RigidTransform,
    DiagramBuilder,
    RollPitchYaw,
    PidController,
    SceneGraph,
    MultibodyPlant,
)

from sim2sim.simulation import BasicSimulator, BasicInnerOnlySimulator
from sim2sim.logging import DynamicLogger
from sim2sim.util import get_parser, create_processed_mesh_directive_str
from sim2sim.images import SphereImageGenerator, NoneImageGenerator
from sim2sim.inverse_graphics import IdentityInverseGraphics
from sim2sim.mesh_processing import IdentityMeshProcessor, QuadricDecimationMeshProcessor
from sim2sim.physical_property_estimator import WaterDensityPhysicalPropertyEstimator, GTPhysicalPropertyEstimator

SCENE_DIRECTIVE = "../../models/table_pid/table_pid_scene_directive.yaml"

# TODO: Allow specifying manipulant with experiment yaml file
MANIPULAND_DIRECTIVE = "../../models/table_pid/table_pid_manipuland_directive.yaml"
MANIPULAND_NAME = "ycb_tomato_soup_can"
MANIPULAND_BASE_LINK_NAME = "ycb_tomato_soup_can_base_link"
MANIPULANT_DEFAULT_POSE = RigidTransform(RollPitchYaw(-np.pi / 2.0, 0.0, 0.0), [0.0, 0.0, 0.57545])  # X_WManipuland

# TODO: Add type info using base classes
LOGGERS = {
    "DynamicLogger": DynamicLogger,
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
}
PHYSICAL_PROPERTY_ESTIMATOR = {
    "WaterDensityPhysicalPropertyEstimator": WaterDensityPhysicalPropertyEstimator,
    "GTPhysicalPropertyEstimator": GTPhysicalPropertyEstimator,
}
SIMULATORS = {
    "BasicSimulator": BasicSimulator,
    "BasicInnerOnlySimulator": BasicInnerOnlySimulator,
}


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


def create_env(
    timestep: float,
    final_table_angle: float,
    no_command_time: float,
    directive_files: List[str] = [],
    directive_strs: List[str] = [],
    manipuland_pose: RigidTransform = MANIPULANT_DEFAULT_POSE,
) -> Tuple[DiagramBuilder, SceneGraph, MultibodyPlant]:
    """Creates the table PID simulation environments without building it."""
    # Create plant
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, timestep)
    parser = get_parser(plant)
    for directive_path in directive_files:
        directive = LoadModelDirectives(directive_path)
        ProcessModelDirectives(directive, parser)
    for directive_str in directive_strs:
        directive = LoadModelDirectivesFromString(directive_str)
        ProcessModelDirectives(directive, parser)

    plant.SetDefaultFreeBodyPose(plant.GetBodyByName(MANIPULAND_BASE_LINK_NAME), manipuland_pose)
    plant.Finalize()

    # Table controller
    pid_controller = builder.AddSystem(PidController(kp=np.array([10.0]), ki=np.array([1.0]), kd=np.array([1.0])))
    pid_controller.set_name("pid_controller")

    # Now "wire up" the controller to the plant
    table_instance = plant.GetModelInstanceByName("table")
    builder.Connect(plant.get_state_output_port(table_instance), pid_controller.get_input_port_estimated_state())
    builder.Connect(pid_controller.get_output_port_control(), plant.get_actuation_input_port(table_instance))

    table_angle_source = builder.AddSystem(TableAngleSource(final_table_angle, no_command_time=no_command_time))
    table_angle_source.set_name("table_angle_source")
    builder.Connect(table_angle_source.get_output_port(), pid_controller.get_input_port_desired_state())

    return builder, scene_graph, plant


def run_table_pid(
    logging_path: str,
    params: dict,
    sim_duration: float,
    timestep: float,
    final_table_angle: float,
    no_command_time: float,
    logging_frequency_hz: float,
):
    """
    Experiment entrypoint for the table PID scene.

    :param logging_path: The path to log the data to.
    :param params: The experiment yaml file dict.
    :param sim_duration: The simulation duration in seconds.
    :param timestep: The timestep to use in seconds.
    :param final_table_angle: The final table angle in radians.
    :param no_command_time: The time before starting the table control in seconds.
    :param logging_frequency_hz: The dynamics logging frequency.
    """

    scene_directive = os.path.join(pathlib.Path(__file__).parent.resolve(), SCENE_DIRECTIVE)
    manipuland_directive = os.path.join(pathlib.Path(__file__).parent.resolve(), MANIPULAND_DIRECTIVE)

    logger_class = LOGGERS[params["logger"]["class"]]
    logger = logger_class(
        logging_frequency_hz=logging_frequency_hz,
        logging_path=logging_path,
        **(params["logger"]["args"] if params["logger"]["args"] is not None else {}),
    )
    logger.log(experiment_description=params)

    # Create folder for temporary files
    tmp_folder = os.path.join(logging_path, "tmp")
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    builder_outer, scene_graph_outer, outer_plant = create_env(
        timestep,
        final_table_angle,
        no_command_time,
        directive_files=[scene_directive, manipuland_directive],
    )

    # Create a new version of the scene for generating camera data
    builder_camera, scene_graph_camera, _ = create_env(
        timestep, final_table_angle, no_command_time, directive_files=[scene_directive, manipuland_directive]
    )
    image_generator_class = IMAGE_GENERATORS[params["image_generator"]["class"]]
    image_generator = image_generator_class(
        builder=builder_camera,
        scene_graph=scene_graph_camera,
        logger=logger,
        **(params["image_generator"]["args"] if params["image_generator"]["args"] is not None else {}),
    )

    images, intrinsics, extrinsics, depths, labels, masks = image_generator.generate_images()
    print("Finished generating images.")

    inverse_graphics_class = INVERSE_GRAPHICS[params["inverse_graphics"]["class"]]
    inverse_graphics = inverse_graphics_class(
        **(params["inverse_graphics"]["args"] if params["inverse_graphics"]["args"] is not None else {}),
        images=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth=depths,
        labels=labels,
        masks=masks,
    )
    raw_mesh, raw_mesh_pose = inverse_graphics.run()
    print("Finished running inverse graphics.")

    mesh_processor_class = MESH_PROCESSORS[params["mesh_processor"]["class"]]
    mesh_processor = mesh_processor_class(
        logger=logger,
        **(params["mesh_processor"]["args"] if params["mesh_processor"]["args"] is not None else {}),
    )
    # TODO: Also support mesh pieces output
    is_primitive, processed_mesh, processed_mesh_piece, primitive_info = mesh_processor.process_mesh(raw_mesh)
    print("Finished mesh processing.")

    # Compute mesh inertia and mass assuming constant density of water
    physical_property_estimator_class = PHYSICAL_PROPERTY_ESTIMATOR[params["physical_property_estimator"]["class"]]
    physical_porperty_estimator = physical_property_estimator_class(
        **(
            params["physical_property_estimator"]["args"]
            if params["physical_property_estimator"]["args"] is not None
            else {}
        ),
    )
    mass, inertia = physical_porperty_estimator.estimate_physical_properties(processed_mesh)
    print("Finished estimating physical properties.")
    logger.log_manipuland_estimated_physics(manipuland_mass_estimated=mass, manipuland_inertia_estimated=inertia)

    # Save mesh data to create SDF files that can be added to a new simulation environment
    logger.log(raw_mesh=raw_mesh, processed_mesh=processed_mesh)
    _, processed_mesh_file_path = logger.save_mesh_data()
    processed_mesh_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "../..", processed_mesh_file_path)

    # Create a directive for processed_mesh manipuland
    processed_mesh_directive = create_processed_mesh_directive_str(
        mass, inertia, processed_mesh_file_path, tmp_folder, "ycb_tomato_soup_can", MANIPULAND_BASE_LINK_NAME
    )

    builder_inner, scene_graph_inner, inner_plant = create_env(
        timestep,
        final_table_angle,
        no_command_time,
        directive_files=[scene_directive],
        directive_strs=[processed_mesh_directive],
        manipuland_pose=RigidTransform(RollPitchYaw(*raw_mesh_pose[:3]), raw_mesh_pose[3:]),
    )

    logger.add_plants(outer_plant, inner_plant)
    logger.add_scene_graphs(scene_graph_outer, scene_graph_inner)

    simulator_class = SIMULATORS[params["simulator"]["class"]]
    simulator = simulator_class(
        outer_builder=builder_outer,
        outer_scene_graph=scene_graph_outer,
        inner_builder=builder_inner,
        inner_scene_graph=scene_graph_inner,
        logger=logger,
        **(params["simulator"]["args"] if params["simulator"]["args"] is not None else {}),
    )
    simulator.simulate(sim_duration)
    print("Finished simulating.")

    logger.save_data()
    print("Finished saving data.")

    # Clean up temporary files
    shutil.rmtree(tmp_folder)
