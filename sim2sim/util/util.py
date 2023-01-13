import os
from typing import List

import open3d as o3d
import trimesh
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
    idx = processed_mesh_file_path.find("sim2sim/")
    procesed_mesh_sdf_path = processed_mesh_file_path[idx + 8 :].replace(".obj", ".sdf")

    with open(procesed_mesh_sdf_path, "w") as f:
        f.write(procesed_mesh_sdf_str)
    print(procesed_mesh_sdf_path)
    return procesed_mesh_sdf_path


def create_decomposition_processed_mesh_sdf_file(
    mass: float,
    inertia: np.ndarray,
    processed_mesh_file_path: str,
    mesh_pieces: List,
    tmp_folder: str,
    manipuland_base_link_name: str,
) -> str:
    """
    Creates and saves an SDF file for the processed decomposition of a mesh.

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
                                <uri>{processed_mesh_file_path.replace("processed_mesh", "raw_mesh")}</uri>
                            </mesh>
                        </geometry>
                    </visual>
    """

    # add the decomposed meshes
    for k, mesh_path in enumerate(mesh_pieces):
        procesed_mesh_sdf_str += f"""
                        <collision name="collision_{k}">
                            <pose>0 0 0 0 0 0</pose>
                            <geometry>
                                <mesh>
                                    <uri>{mesh_path}</uri>
                                </mesh>
                            </geometry>
                        </collision>
        """
    procesed_mesh_sdf_str += f"""
                </link>
            </model>
        </sdf>
    """
    # hack now, don't wanna save the important sdf file to tmp
    idx = processed_mesh_file_path.find("sim2sim/")
    procesed_mesh_sdf_path = processed_mesh_file_path[idx + 8 :].replace(".obj", ".sdf")

    # os.path.join(tmp_folder, "processed_mesh.sdf")
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
    if not (processed_mesh_file_path).endswith(".obj"):
        dir_name = "/".join(processed_mesh_file_path.split("/")[:-1])
        listed_files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if "piece" in f]
        procesed_mesh_sdf_path = create_decomposition_processed_mesh_sdf_file(
            mass, inertia, processed_mesh_file_path + ".obj", listed_files, tmp_folder, manipuland_base_link_name
        )
    else:
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


def create_masked_images(input_path: str) -> str:
    """
    Creates a sequence of masked image of interest.

    :param input_path: The folder directory for both RGB and binary mask folder.
    :return output path of the masked images.
    """
    sample_path = input_path.rsplit("/", 1)[0]
    lst = os.listdir(sample_path + "/images/")
    num_files = len(lst)

    for i in range(num_files):
        if i < 10:
            j = "0" + str(i)
        else:
            j = i
        image_path = sample_path + f"/images/image00{j}.png"
        mask_path = sample_path + f"/binary_masks/mask00{j}.png"
        colourRaw = np.asarray(o3d.io.read_image(image_path))
        maskImg = np.asarray(o3d.io.read_image(mask_path))
        if np.max(maskImg) > 0:
            maskImg = np.where(maskImg > 0, 1, maskImg)
        mask_3d = np.stack((maskImg, maskImg, maskImg), axis=2)
        input_rgb = colourRaw * mask_3d
        masked_image = o3d.geometry.Image(input_rgb)
        output_path = sample_path + f"/masked_images/image00{j}.png"
        o3d.io.write_image(output_path, masked_image)
    print(f"Successfully masked images at dir: {sample_path + '/masked_images'}")
    return sample_path + "/masked_images"


def open3d_to_trimesh(src: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """
    Convert mesh from open3d to trimesh
    https://github.com/wkentaro/morefusion/blob/b8b892b3fbc384982a4929b1418ee29393069b11/morefusion/utils/open3d_to_trimesh.py
    """
    if isinstance(src, o3d.geometry.TriangleMesh):
        vertex_colors = None
        if src.has_vertex_colors:
            vertex_colors = np.asarray(src.vertex_colors)
        dst = trimesh.Trimesh(
            vertices=np.asarray(src.vertices),
            faces=np.asarray(src.triangles),
            vertex_normals=np.asarray(src.vertex_normals),
            vertex_colors=vertex_colors,
        )
    else:
        raise ValueError("Unsupported type of src: {}".format(type(src)))

    return dst
