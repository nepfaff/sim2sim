import os
import pathlib
from typing import Callable, List

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
    MeshProcessorResult,
)
from sim2sim.images import (
    SphereImageGenerator,
    NoneImageGenerator,
    IIWAWristSphereImageGenerator,
)
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
    IdentityVTKMeshProcessor,
    IdentityVTKPiecesMeshProcessor,
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
    "IIWAWristSphereImageGenerator": IIWAWristSphereImageGenerator,
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
    "IdentityVTKMeshProcessor": IdentityVTKMeshProcessor,
    "IdentityVTKPiecesMeshProcessor": IdentityVTKPiecesMeshProcessor,
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
    manipuland_base_link_names: List[str],
    manipuland_default_poses: List[RigidTransform],
    hydroelastic_manipuland: bool,
    scene_directive_path: str,
    manipuland_directive_paths: List[str],
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
        manipuland_base_link_names=manipuland_base_link_names,
        manipuland_poses=manipuland_default_poses,
        hydroelastic_manipuland=hydroelastic_manipuland,
        directive_files=[scene_directive_path, *manipuland_directive_paths],
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
    raw_meshes, raw_mesh_poses = inverse_graphics.run()
    # TODO: Log 'raw_mesh_poses' and 'manipuland_default_pose_transforms' as meta-data
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
    mesh_processor_results: List[MeshProcessorResult] = mesh_processor.process_meshes(
        raw_meshes
    )
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
    physical_properties = physical_porperty_estimator.estimate_physical_properties(
        raw_meshes
    )
    print(
        f"Finished estimating physical properties{f' for {prefix}' if prefix else ''}."
    )
    logger.log(manipuland_physical_properties=physical_properties)

    # Save mesh data to create SDF files that can be added to a new simulation environment
    # Only save the raw mesh if we use it for visualization
    if not logger.is_kProximity:
        logger.log(raw_meshes=raw_meshes)
    logger.log(mesh_processor_results=mesh_processor_results)
    raw_mesh_file_paths, processed_mesh_file_paths = logger.save_mesh_data(
        prefix=prefix
    )
    raw_mesh_file_paths = [
        (
            os.path.join(pathlib.Path(__file__).parent.resolve(), "../..", path)
            if not logger.is_kProximity
            else None
        )
        for path in raw_mesh_file_paths
    ]
    processed_mesh_file_paths = [
        os.path.join(pathlib.Path(__file__).parent.resolve(), "../..", path)
        for path in processed_mesh_file_paths
    ]

    # Create directives for the processed_mesh manipulands
    processed_mesh_directives = []
    for (
        i,
        (
            manipuland_base_link_name,
            mesh_processor_result,
            physical_property,
            raw_mesh_file_path,
            processed_mesh_file_path,
        ),
    ) in enumerate(
        zip(
            manipuland_base_link_names,
            mesh_processor_results,
            physical_properties,
            (
                [None] * len(manipuland_base_link_names)
                if len(raw_mesh_file_paths) == 0
                else raw_mesh_file_paths
            ),
            processed_mesh_file_paths,
        )
    ):
        manipuland_name = manipuland_base_link_name.replace("_base_link", "")
        if mesh_processor_result.result_type == MeshProcessorResult.ResultType.SDF_PATH:
            processed_mesh_directive = create_directive_str_for_sdf_path(
                mesh_processor_result.get_result(), manipuland_name
            )
        elif (
            mesh_processor_result.result_type
            == MeshProcessorResult.ResultType.PRIMITIVE_INFO
        ):
            processed_mesh_directive = create_processed_mesh_primitive_directive_str(
                mesh_processor_result.get_result(),
                physical_property,
                logger._mesh_dir_path,
                manipuland_name,
                manipuland_base_link_name,
                hydroelastic=hydroelastic_manipuland,
                prefix=prefix,
                idx=i,
                visual_mesh_file_path=raw_mesh_file_path,
            )
        else:
            processed_mesh_directive = create_processed_mesh_directive_str(
                physical_property,
                processed_mesh_file_path,
                logger._mesh_dir_path,
                manipuland_name,
                manipuland_base_link_name,
                hydroelastic=hydroelastic_manipuland,
                prefix=prefix,
                idx=i,
                visual_mesh_file_path=raw_mesh_file_path,
            )
        processed_mesh_directives.append(processed_mesh_directive)

    manipuland_poses = [
        RigidTransform(RollPitchYaw(*pose[:3]), pose[3:]) for pose in raw_mesh_poses
    ]
    builder, scene_graph, plant = create_env_func(
        timestep=timestep,
        env_params=params[f"{prefix}env"],
        manipuland_base_link_names=manipuland_base_link_names,
        hydroelastic_manipuland=hydroelastic_manipuland,
        directive_files=[scene_directive_path],
        directive_strs=processed_mesh_directives,
        manipuland_poses=manipuland_poses,
        **kwargs,
    )

    return builder, scene_graph, plant


def run_experiment(
    logging_path: str,
    params: dict,
    sim_duration: float,
    timestep: float,
    logging_frequency_hz: float,
    manipuland_directives: List[str],
    scene_directive: str,
    manipuland_base_link_names: List[str],
    manipuland_default_poses: List[List[float]],
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
    :param manipuland_directives: The file paths of the outer manipuland directives. The
        paths should be relative to this script.
    :param scene_directive: The file path of the scene directive. The path should be
        relative to this script.
    :param manipuland_base_link_names: The base link names of the outer manipulands.
    :param manipuland_default_poses: The default poses of the outer manipulands of form
        [roll, pitch, yaw, x, y, z].
    :param hydroelastic_manipuland: Whether to use hydroelastic or point contact for the
        inner manipuland.
    :param is_pipeline_comparison: Whether it is a sim2sim pipeline comparison
        experiment.
    :param create_env_func: The experiment specific setup function that creates the
        Drake environment/ system.
    """
    assert (
        len(manipuland_directives)
        == len(manipuland_default_poses)
        == len(manipuland_base_link_names)
    ), (
        "The number of manipuland directives must equal the number of manipuland "
        + "default poses and the number of manipuland base link names!"
    )

    scene_directive_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), scene_directive
    )
    manipuland_directive_paths = [
        os.path.join(pathlib.Path(__file__).parent.resolve(), directive)
        for directive in manipuland_directives
    ]

    logger_class = LOGGERS[params["logger"]["class"]]
    logger = logger_class(
        logging_frequency_hz=logging_frequency_hz,
        logging_path=logging_path,
        **(params["logger"]["args"] if params["logger"]["args"] is not None else {}),
    )
    logger.log(experiment_description=params)

    manipuland_default_poses = [
        RigidTransform(RollPitchYaw(*pose[:3]), pose[3:])
        for pose in manipuland_default_poses
    ]
    if is_pipeline_comparison:
        outer_builder, outer_scene_graph, outer_plant = run_pipeline(
            prefix="outer",
            params=params,
            logger=logger,
            timestep=timestep,
            manipuland_base_link_names=manipuland_base_link_names,
            manipuland_default_poses=manipuland_default_poses,
            hydroelastic_manipuland=hydroelastic_manipuland,
            scene_directive_path=scene_directive_path,
            manipuland_directive_paths=manipuland_directive_paths,
            create_env_func=create_env_func,
            **kwargs,
        )

        inner_builder, inner_scene_graph, inner_plant = run_pipeline(
            prefix="inner",
            params=params,
            logger=logger,
            timestep=timestep,
            manipuland_base_link_names=manipuland_base_link_names,
            manipuland_default_poses=manipuland_default_poses,
            hydroelastic_manipuland=hydroelastic_manipuland,
            scene_directive_path=scene_directive_path,
            manipuland_directive_paths=manipuland_directive_paths,
            create_env_func=create_env_func,
            **kwargs,
        )
    else:
        outer_builder, outer_scene_graph, outer_plant = create_env_func(
            env_params=params["env"],
            timestep=timestep,
            manipuland_base_link_names=manipuland_base_link_names,
            manipuland_poses=manipuland_default_poses,
            hydroelastic_manipuland=hydroelastic_manipuland,
            directive_files=[scene_directive_path, *manipuland_directive_paths],
            **kwargs,
        )

        inner_builder, inner_scene_graph, inner_plant = run_pipeline(
            params=params,
            logger=logger,
            timestep=timestep,
            manipuland_base_link_names=manipuland_base_link_names,
            manipuland_default_poses=manipuland_default_poses,
            hydroelastic_manipuland=hydroelastic_manipuland,
            scene_directive_path=scene_directive_path,
            manipuland_directive_paths=manipuland_directive_paths,
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
