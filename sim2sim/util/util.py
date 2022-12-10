import os
from typing import List

import numpy as np
from pydrake.all import MultibodyPlant, Parser, RigidTransform
from manipulation.scenarios import AddPackagePaths
from manipulation.meshcat_utils import AddMeshcatTriad


def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.package_map().Add("sim2sim", os.path.abspath(""))
    return parser


def visualize_poses(poses: List[RigidTransform], meshcat) -> None:
    for i, pose in enumerate(poses):
        AddMeshcatTriad(meshcat, f"pose{i}", length=0.15, radius=0.006, X_PT=pose)


def create_processed_mesh_sdf_file(
    mass: float, inertia: np.ndarray, processed_mesh_file_path: str, tmp_folder: str, manipuland_base_link_name: str
) -> str:
    """
    Creates and saves an SDF file for the processed mesh.

    :param mass: The object mass in kg.
    :param inertia: The moment of inertia matrix of shape (3,3).
    :param processed_mesh_file_path: The path to the processed mesh obj file.
    :param tmp_folder: The folder to write the sdf file to.
    :return procesed_mesh_sdf_path: The path to the SDF file.
    """
    procesed_mesh_sdf_str = f"""
        <?xml version="1.0"?>
        <sdf version="1.7">
            <model name="processed_manipuland_mesh">
                <link name="{manipuland_base_link_name}">
                    <inertial>
                        <inertia>
                            <ixx>{inertia[0,0]}</ixx>
                            <ixy>{inertia[0,1]}</ixy>
                            <ixz>{inertia[0,2]}</ixz>
                            <iyy>{inertia[1,1]}</iyy>
                            <iyz>{inertia[1,2]}</iyz>
                            <izz>{inertia[2,2]}</izz>
                        </inertia>
                        <mass>{mass}</mass>
                    </inertial>

                    <visual name="visual">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{processed_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
                    </visual>
                    <collision name="collision">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{processed_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
                        <drake:proximity_properties>
                            <drake:rigid_hydroelastic/>
                        </drake:proximity_properties>
                    </collision>
                </link>
            </model>
        </sdf>
    """
    procesed_mesh_sdf_path = os.path.join(tmp_folder, "processed_mesh.sdf")
    with open(procesed_mesh_sdf_path, "w") as f:
        f.write(procesed_mesh_sdf_str)

    return procesed_mesh_sdf_path


def create_processed_mesh_directive_str(
    mass: float,
    inertia: np.ndarray,
    processed_mesh_file_path: str,
    tmp_folder: str,
    model_name: str,
    manipuland_base_link_name: str,
) -> str:
    """
    Creates a directive for the processed mesh.

    :param mass: The object mass in kg.
    :param inertia: The moment of inertia matrix of shape (3,3).
    :param processed_mesh_file_path: The path to the processed mesh obj file.
    :param tmp_folder: The folder to write the sdf file to.
    :return processed_mesh_directive_str: The directive string for the processed mesh.
    """
    procesed_mesh_sdf_path = create_processed_mesh_sdf_file(
        mass, inertia, processed_mesh_file_path, tmp_folder, manipuland_base_link_name
    )
    processed_mesh_directive = f"""
        directives:
        - add_model:
            name: {model_name}
            file: package://sim2sim/{procesed_mesh_sdf_path}
    """
    return processed_mesh_directive
