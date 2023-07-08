import os
import pathlib
from typing import Callable

from pydrake.all import RigidTransform, RollPitchYaw

from sim2sim.simulation import (
    BasicSimulator,
    BasicInnerOnlySimulator,
    EquationErrorBasicSimulator,
    PlanarPushingSimulator,
    EquationErrorPlanarPushingSimulator,
    RandomForceSimulator,
    IIWARearrangementSimulator,
    IIWAPushInHoleSimulator,
)
from sim2sim.logging import DynamicLogger, PlanarPushingLogger
from sim2sim.util import (
    create_processed_mesh_directive_str,
    create_processed_mesh_primitive_directive_str,
    create_directive_str_for_sdf_path,
)
from sim2sim.images import SphereImageGenerator, NoneImageGenerator
from sim2sim.inverse_graphics import IdentityInverseGraphics
from sim2sim.mesh_processing import (
    IdentityMeshProcessor,
    QuadricDecimationMeshProcessor,
    SphereMeshProcessor,
    GMMMeshProcessor,
    ConvexDecompMeshProcessor,
    CoACDMeshProcessor,
    FuzzyMetaballMeshProcessor,
    IdentityPrimitiveMeshProcessor,
    IdentitySDFMeshProcessor,
    IdentityMeshPiecesMeshProcessor,
)
from sim2sim.physical_property_estimator import (
    WaterDensityPhysicalPropertyEstimator,
    GTPhysicalPropertyEstimator,
)


# TODO: Add type info using base classes
LOGGERS = {
    "DynamicLogger": DynamicLogger,
    "PlanarPushingLogger": PlanarPushingLogger,
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
    "SphereMeshProcessor": SphereMeshProcessor,
    "GMMMeshProcessor": GMMMeshProcessor,
    "ConvexDecompMeshProcessor": ConvexDecompMeshProcessor,
    "CoACDMeshProcessor": CoACDMeshProcessor,
    "FuzzyMetaballMeshProcessor": FuzzyMetaballMeshProcessor,
    "IdentityPrimitiveMeshProcessor": IdentityPrimitiveMeshProcessor,
    "IdentitySDFMeshProcessor": IdentitySDFMeshProcessor,
    "IdentityMeshPiecesMeshProcessor": IdentityMeshPiecesMeshProcessor,
}
PHYSICAL_PROPERTY_ESTIMATOR = {
    "WaterDensityPhysicalPropertyEstimator": WaterDensityPhysicalPropertyEstimator,
    "GTPhysicalPropertyEstimator": GTPhysicalPropertyEstimator,
}
SIMULATORS = {
    "BasicSimulator": BasicSimulator,
    "BasicInnerOnlySimulator": BasicInnerOnlySimulator,
    "EquationErrorBasicSimulator": EquationErrorBasicSimulator,
    "PlanarPushingSimulator": PlanarPushingSimulator,
    "EquationErrorPlanarPushingSimulator": EquationErrorPlanarPushingSimulator,
    "RandomForceSimulator": RandomForceSimulator,
    "IIWARearrangementSimulator": IIWARearrangementSimulator,
    "IIWAPushInHoleSimulator": IIWAPushInHoleSimulator,
}


def run_pipeline(
    params: dict,
    logger: PlanarPushingLogger,
    timestep: float,
    manipuland_base_link_name: str,
    manipuland_default_pose: RigidTransform,
    hydroelastic_manipuland: bool,
    scene_directive_path: str,
    manipuland_directive_path: str,
    create_env_func: Callable,
    prefix: str = "",
    **kwargs,
):
    """
    Runs the sim2sim pipeline of camera data generation, mesh generation, mesh
    processing, and physical property estimation.

    :param prefix: The prefix of the pipeline components in `params`.
    """
    prefix = prefix + "_" if prefix else ""

    # Create a new version of the scene for generating camera data
    camera_builder, camera_scene_graph, _ = create_env_func(
        env_params=params[f"{prefix}env"],
        timestep=timestep,
        manipuland_base_link_name=manipuland_base_link_name,
        manipuland_pose=manipuland_default_pose,
        hydroelastic_manipuland=hydroelastic_manipuland,
        directive_files=[scene_directive_path, manipuland_directive_path],
        **kwargs,
    )
    image_generator_name = f"{prefix}image_generator"
    image_generator_class = IMAGE_GENERATORS[params[image_generator_name]["class"]]
    image_generator = image_generator_class(
        builder=camera_builder,
        scene_graph=camera_scene_graph,
        logger=logger,
        **(
            params[image_generator_name]["args"]
            if params[image_generator_name]["args"] is not None
            else {}
        ),
    )

    (
        images,
        intrinsics,
        extrinsics,
        depths,
        labels,
        masks,
    ) = image_generator.generate_images()
    print(f"Finished generating images{f' for {prefix}' if prefix else ''}.")

    inverse_graphics_name = f"{prefix}inverse_graphics"
    inverse_graphics_class = INVERSE_GRAPHICS[params[inverse_graphics_name]["class"]]
    inverse_graphics = inverse_graphics_class(
        **(
            params[inverse_graphics_name]["args"]
            if params[inverse_graphics_name]["args"] is not None
            else {}
        ),
        images=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth=depths,
        labels=labels,
        masks=masks,
    )
    raw_mesh, raw_mesh_pose = inverse_graphics.run()
    # TODO: Log 'raw_mesh_pose' and 'manipuland_default_pose_transform' as meta-data
    print(f"Finished running inverse graphics{f' for {prefix}' if prefix else ''}.")

    mesh_processor_name = f"{prefix}mesh_processor"
    mesh_processor_class = MESH_PROCESSORS[params[mesh_processor_name]["class"]]
    mesh_processor = mesh_processor_class(
        logger=logger,
        **(
            params[mesh_processor_name]["args"]
            if params[mesh_processor_name]["args"] is not None
            else {}
        ),
    )
    (
        is_primitive,
        processed_mesh,
        processed_mesh_pieces,
        primitive_info,
        mesh_sdf_path,
    ) = mesh_processor.process_mesh(raw_mesh)
    print(f"Finished mesh processing{f' for {prefix}' if prefix else ''}.")

    # Compute mesh inertia and mass
    physical_porperty_estimator_name = f"{prefix}physical_property_estimator"
    physical_property_estimator_class = PHYSICAL_PROPERTY_ESTIMATOR[
        params[physical_porperty_estimator_name]["class"]
    ]
    physical_porperty_estimator = physical_property_estimator_class(
        **(
            params[physical_porperty_estimator_name]["args"]
            if params[physical_porperty_estimator_name]["args"] is not None
            else {}
        ),
    )
    (
        mass,
        inertia,
        center_of_mass,
    ) = physical_porperty_estimator.estimate_physical_properties(processed_mesh)
    print(
        f"Finished estimating physical properties{f' for {prefix}' if prefix else ''}."
    )
    logger.log_manipuland_estimated_physics(
        manipuland_mass_estimated=mass,
        manipuland_inertia_estimated=inertia,
        manipuland_com_estimated=center_of_mass,
    )

    # Save mesh data to create SDF files that can be added to a new simulation environment
    # Only save the raw mesh if we use it for visualization
    if not logger.is_kProximity:
        logger.log(raw_mesh=raw_mesh)
    logger.log(
        processed_mesh=processed_mesh, processed_mesh_piece=processed_mesh_pieces
    )
    raw_mesh_file_path, processed_mesh_file_path = logger.save_mesh_data(prefix=prefix)
    raw_mesh_file_path = (
        os.path.join(
            pathlib.Path(__file__).parent.resolve(), "../..", raw_mesh_file_path
        )
        if not logger.is_kProximity
        else None
    )
    processed_mesh_file_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "../..", processed_mesh_file_path
    )

    # Create a directive for processed_mesh manipuland
    if mesh_sdf_path is not None:
        processed_mesh_directive = create_directive_str_for_sdf_path(
            mesh_sdf_path, params[f"{prefix}env"]["obj_name"]
        )
    elif is_primitive:
        processed_mesh_directive = create_processed_mesh_primitive_directive_str(
            primitive_info,
            mass,
            inertia,
            center_of_mass,
            logger._mesh_dir_path,
            params[f"{prefix}env"]["obj_name"],
            manipuland_base_link_name,
            hydroelastic=hydroelastic_manipuland,
            prefix=prefix,
            visual_mesh_file_path=raw_mesh_file_path,
        )
    else:
        processed_mesh_directive = create_processed_mesh_directive_str(
            mass,
            inertia,
            center_of_mass,
            processed_mesh_file_path,
            logger._mesh_dir_path,
            params[f"{prefix}env"]["obj_name"],
            manipuland_base_link_name,
            hydroelastic=hydroelastic_manipuland,
            prefix=prefix,
            visual_mesh_file_path=raw_mesh_file_path,
        )

    builder, scene_graph, plant = create_env_func(
        timestep=timestep,
        env_params=params[f"{prefix}env"],
        manipuland_base_link_name=manipuland_base_link_name,
        hydroelastic_manipuland=hydroelastic_manipuland,
        directive_files=[scene_directive_path],
        directive_strs=[processed_mesh_directive],
        manipuland_pose=RigidTransform(
            RollPitchYaw(*raw_mesh_pose[:3]), raw_mesh_pose[3:]
        ),
        **kwargs,
    )

    return builder, scene_graph, plant


def run_experiment(
    logging_path: str,
    params: dict,
    sim_duration: float,
    timestep: float,
    logging_frequency_hz: float,
    manipuland_directive: str,
    scene_directive: str,
    manipuland_base_link_name: str,
    manipuland_default_pose: str,
    hydroelastic_manipuland: bool,
    is_pipeline_comparison: bool,
    create_env_func: Callable,
    **kwargs,
):
    """
    Experiment entrypoint.

    :param logging_path: The path to log the data to.
    :param params: The experiment yaml file dict.
    :param sim_duration: The simulation duration in seconds.
    :param timestep: The timestep to use in seconds.
    :param logging_frequency_hz: The dynamics logging frequency.
    :param manipuland_directive: The file path of the outer manipuland directive. The
        path should be relative to this script.
    :param scene_directive: The file path of the scene directive. The path should be
        relative to this script.
    :param manipuland_base_link_name: The base link name of the outer manipuland.
    :param manipuland_default_pose: The default pose of the outer manipuland of form
        [roll, pitch, yaw, x, y, z].
    :param hydroelastic_manipuland: Whether to use hydroelastic or point contact for the
        inner manipuland.
    :param is_pipeline_comparison: Whether it is a sim2sim pipeline comparison
        experiment.
    :param create_env_func: The experiment specific setup function that creates the
        Drake environment/ system.
    """
    scene_directive_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), scene_directive
    )
    manipuland_directive_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), manipuland_directive
    )

    logger_class = LOGGERS[params["logger"]["class"]]
    logger = logger_class(
        logging_frequency_hz=logging_frequency_hz,
        logging_path=logging_path,
        **(params["logger"]["args"] if params["logger"]["args"] is not None else {}),
    )
    logger.log(experiment_description=params)

    manipuland_default_pose = RigidTransform(
        RollPitchYaw(*manipuland_default_pose[:3]), manipuland_default_pose[3:]
    )
    if is_pipeline_comparison:
        outer_builder, outer_scene_graph, outer_plant = run_pipeline(
            prefix="outer",
            params=params,
            logger=logger,
            timestep=timestep,
            manipuland_base_link_name=manipuland_base_link_name,
            manipuland_default_pose=manipuland_default_pose,
            hydroelastic_manipuland=hydroelastic_manipuland,
            scene_directive_path=scene_directive_path,
            manipuland_directive_path=manipuland_directive_path,
            create_env_func=create_env_func,
            **kwargs,
        )

        inner_builder, inner_scene_graph, inner_plant = run_pipeline(
            prefix="inner",
            params=params,
            logger=logger,
            timestep=timestep,
            manipuland_base_link_name=manipuland_base_link_name,
            manipuland_default_pose=manipuland_default_pose,
            hydroelastic_manipuland=hydroelastic_manipuland,
            scene_directive_path=scene_directive_path,
            manipuland_directive_path=manipuland_directive_path,
            create_env_func=create_env_func,
            **kwargs,
        )
    else:
        outer_builder, outer_scene_graph, outer_plant = create_env_func(
            env_params=params["env"],
            timestep=timestep,
            manipuland_base_link_name=manipuland_base_link_name,
            manipuland_pose=manipuland_default_pose,
            hydroelastic_manipuland=hydroelastic_manipuland,
            directive_files=[scene_directive_path, manipuland_directive_path],
            **kwargs,
        )

        inner_builder, inner_scene_graph, inner_plant = run_pipeline(
            params=params,
            logger=logger,
            timestep=timestep,
            manipuland_base_link_name=manipuland_base_link_name,
            manipuland_default_pose=manipuland_default_pose,
            hydroelastic_manipuland=hydroelastic_manipuland,
            scene_directive_path=scene_directive_path,
            manipuland_directive_path=manipuland_directive_path,
            create_env_func=create_env_func,
            **kwargs,
        )

    logger.add_plants(outer_plant, inner_plant)
    logger.add_scene_graphs(outer_scene_graph, inner_scene_graph)

    simulator_class = SIMULATORS[params["simulator"]["class"]]
    simulator = simulator_class(
        outer_builder=outer_builder,
        outer_scene_graph=outer_scene_graph,
        inner_builder=inner_builder,
        inner_scene_graph=inner_scene_graph,
        logger=logger,
        is_hydroelastic=params[f"{'inner_' if is_pipeline_comparison else ''}env"][
            "contact_model"
        ]
        != "point",  # Visualize outer contact forces if inner/outer use different contact engines
        **(
            params["simulator"]["args"]
            if params["simulator"]["args"] is not None
            else {}
        ),
    )
    simulator.simulate(sim_duration)
    print("Finished simulating.")

    logger.save_data()
    print("Finished saving data.")
