import os
from typing import List, Dict, Any, Optional
import copy
import shutil
from pathlib import Path
import subprocess

import open3d as o3d
import trimesh
import numpy as np
import meshio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pydrake.all import (
    MultibodyPlant,
    Parser,
    RigidTransform,
    ContactVisualizerParams,
    ProximityProperties,
    Quaternion,
    Shape,
    Box,
    Cylinder,
    Sphere,
    Capsule,
    CoulombFriction,
    SpatialInertia,
    UnitInertia,
)
from manipulation.utils import AddPackagePaths
from manipulation.meshcat_utils import AddMeshcatTriad

from .dataclasses import PhysicalProperties


def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.package_map().Add("sim2sim", os.path.abspath(""))
    return parser


def visualize_poses(poses: List[RigidTransform], meshcat) -> None:
    for i, pose in enumerate(poses):
        AddMeshcatTriad(meshcat, f"pose{i}", length=0.15, radius=0.006, X_PT=pose)


def construct_drake_proximity_properties_sdf_str(
    physical_properties: PhysicalProperties, is_hydroelastic: bool
) -> str:
    """
    Constructs a Drake proximity properties SDF string using the proximity properties
    contained in `physical_properties`. Only adds the Hydroelastic properties if
    `is_hydroelastic` is true.
    """
    proximity_properties_str = """
            <drake:proximity_properties>
        """
    if is_hydroelastic:
        if physical_properties.is_compliant:
            assert (
                physical_properties.hydroelastic_modulus is not None
            ), "Require a Hydroelastic modulus for compliant Hydroelastic objects!"
            proximity_properties_str += f"""
                        <drake:compliant_hydroelastic/>
                        <drake:hydroelastic_modulus>
                            {physical_properties.hydroelastic_modulus}
                        </drake:hydroelastic_modulus>
                """
        else:
            proximity_properties_str += """
                    <drake:rigid_hydroelastic/>
            """
        if physical_properties.mesh_resolution_hint is not None:
            proximity_properties_str += f"""
                    <drake:mesh_resolution_hint>
                        {physical_properties.mesh_resolution_hint}
                    </drake:mesh_resolution_hint>
            """
    if physical_properties.hunt_crossley_dissipation is not None:
        proximity_properties_str += f"""
                    <drake:hunt_crossley_dissipation>
                        {physical_properties.hunt_crossley_dissipation}
                    </drake:hunt_crossley_dissipation>
            """
    if physical_properties.mu_dynamic is not None:
        proximity_properties_str += f"""
                    <drake:mu_dynamic>
                        {physical_properties.mu_dynamic}
                    </drake:mu_dynamic>
            """
    if physical_properties.mu_static is not None:
        proximity_properties_str += f"""
                    <drake:mu_static>
                        {physical_properties.mu_static}
                    </drake:mu_static>
            """
    proximity_properties_str += """
            </drake:proximity_properties>
        """
    return proximity_properties_str


def create_processed_mesh_sdf_file(
    physical_properties: PhysicalProperties,
    processed_mesh_file_path: str,
    tmp_folder: str,
    manipuland_base_link_name: str,
    is_hydroelastic: bool,
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates and saves an SDF file for the processed mesh.

    :param physical_properties: The physical properties.
    :param processed_mesh_file_path: The path to the processed mesh obj/vtk file.
    :param tmp_folder: The folder to write the sdf file to.
    :param is_hydroelastic: Whether to make the body rigid hydroelastic.
    :param visual_mesh_file_path: The path to the mesh to use for the visual geometry.
    :return procesed_mesh_sdf_path: The path to the SDF file.
    """
    com = physical_properties.center_of_mass
    procesed_mesh_sdf_str = f"""
        <?xml version="1.0"?>
        <sdf version="1.7">
            <model name="processed_manipuland_mesh">
                <link name="{manipuland_base_link_name}">
                    <inertial>
                        <inertia>
                            <ixx>{physical_properties.inertia[0,0]}</ixx>
                            <ixy>{physical_properties.inertia[0,1]}</ixy>
                            <ixz>{physical_properties.inertia[0,2]}</ixz>
                            <iyy>{physical_properties.inertia[1,1]}</iyy>
                            <iyz>{physical_properties.inertia[1,2]}</iyz>
                            <izz>{physical_properties.inertia[2,2]}</izz>
                        </inertia>
                        <mass>{physical_properties.mass}</mass>
                        <pose>{com[0]} {com[1]} {com[2]} 0 0 0</pose>
                    </inertial>
        """

    if visual_mesh_file_path is not None:
        procesed_mesh_sdf_str += f"""
                    <visual name="visual">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{visual_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
                    </visual>
            """

    procesed_mesh_sdf_str += f"""
                    <collision name="collision">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{processed_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
        """

    procesed_mesh_sdf_str += construct_drake_proximity_properties_sdf_str(
        physical_properties, is_hydroelastic
    )

    procesed_mesh_sdf_str += """
                    </collision>
                    </link>
                </model>
            </sdf>
        """

    idx = processed_mesh_file_path.find("sim2sim/")
    procesed_mesh_sdf_path = processed_mesh_file_path[idx + 8 : -3] + "sdf"

    with open(procesed_mesh_sdf_path, "w") as f:
        f.write(procesed_mesh_sdf_str)
    return procesed_mesh_sdf_path


def create_decomposition_processed_mesh_sdf_file(
    physical_properties: PhysicalProperties,
    mesh_pieces: List[str],
    sdf_folder: str,
    manipuland_base_link_name: str,
    is_hydroelastic: bool,
    prefix: str = "",
    idx: int = 0,
    parts_are_convex: bool = True,
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates and saves an SDF file for the processed decomposition of a mesh.

    :param physical_properties: The physical properties.
    :param mesh_pieces: A list of mesh piece paths.
    :param sdf_folder: The folder to write the sdf file to.
    :param is_hydroelastic: Whether to make the body rigid hydroelastic.
    :param prefix: An optional prefix for the processed mesh sdf file name.
    :param idx: An optional index for the processed mesh sdf file name.
    :param visual_mesh_file_path: The path to the mesh to use for the visual geometry.
    """
    com = physical_properties.center_of_mass
    procesed_mesh_sdf_str = f"""
        <?xml version="1.0"?>
        <sdf version="1.7">
            <model name="processed_manipuland_mesh">
                <link name="{manipuland_base_link_name}">
                    <inertial>
                        <inertia>
                            <ixx>{physical_properties.inertia[0,0]}</ixx>
                            <ixy>{physical_properties.inertia[0,1]}</ixy>
                            <ixz>{physical_properties.inertia[0,2]}</ixz>
                            <iyy>{physical_properties.inertia[1,1]}</iyy>
                            <iyz>{physical_properties.inertia[1,2]}</iyz>
                            <izz>{physical_properties.inertia[2,2]}</izz>
                        </inertia>
                        <mass>{physical_properties.mass}</mass>
                        <pose>{com[0]} {com[1]} {com[2]} 0 0 0</pose>
                    </inertial>
        """

    if visual_mesh_file_path is not None:
        procesed_mesh_sdf_str += f"""
                    <visual name="visual">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{visual_mesh_file_path}</uri>
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
                                    {'<drake:declare_convex/>' if parts_are_convex else ''}
                                    <uri>{mesh_path}</uri>
                                </mesh>
                            </geometry>
            """

        procesed_mesh_sdf_str += construct_drake_proximity_properties_sdf_str(
            physical_properties, is_hydroelastic
        )

        procesed_mesh_sdf_str += """
                </collision>
            """

    procesed_mesh_sdf_str += """
                    </link>
                </model>
            </sdf>
        """

    procesed_mesh_sdf_path = os.path.join(
        sdf_folder, f"{prefix}processed_mesh_{idx}.sdf"
    )
    with open(procesed_mesh_sdf_path, "w") as f:
        f.write(procesed_mesh_sdf_str)

    return procesed_mesh_sdf_path


def create_processed_mesh_directive_str(
    physical_properties: PhysicalProperties,
    processed_mesh_file_path: str,
    sdf_folder: str,
    model_name: str,
    manipuland_base_link_name: str,
    hydroelastic: bool,
    prefix: str = "",
    idx: int = 0,
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates a directive for the processed mesh.

    :param physical_properties: The physical properties.
    :param processed_mesh_file_path: The path to the processed mesh pieces directory,
        obj file, or vtk file.
    :param sdf_folder: The folder to write the sdf file to.
    :param model_name: The name of the directive model.
    :param hydroelastic: Whether to make the body rigid hydroelastic.
    :param prefix: An optional prefix for the processed mesh sdf file name.
    :param idx: An optional index for the processed mesh sdf file name.
    :param visual_mesh_file_path: The path to the mesh to use for the visual geometry.
    """
    if os.path.isdir(processed_mesh_file_path):
        listed_files = [
            os.path.join(processed_mesh_file_path, f)
            for f in os.listdir(processed_mesh_file_path)
            if "mesh_piece_" in f
        ]
        # Assume that the individual pieces are convex
        # Cannot declare VTK files as convex
        procesed_mesh_sdf_path = create_decomposition_processed_mesh_sdf_file(
            physical_properties,
            listed_files,
            sdf_folder,
            manipuland_base_link_name,
            hydroelastic,
            prefix,
            idx,
            parts_are_convex=listed_files[0][-3:].lower() == "obj",
            visual_mesh_file_path=visual_mesh_file_path,
        )
    else:
        procesed_mesh_sdf_path = create_processed_mesh_sdf_file(
            physical_properties,
            processed_mesh_file_path,
            sdf_folder,
            manipuland_base_link_name,
            hydroelastic,
            visual_mesh_file_path=visual_mesh_file_path,
        )
    processed_mesh_directive = f"""
        directives:
        - add_model:
            name: {model_name}
            file: package://sim2sim/{procesed_mesh_sdf_path}
    """
    return processed_mesh_directive


def create_processed_mesh_primitive_sdf_file(
    primitive_info: List[Dict[str, Any]],
    physical_properties: PhysicalProperties,
    sdf_folder: str,
    manipuland_base_link_name: str,
    is_hydroelastic: bool,
    prefix: str = "",
    idx: int = 0,
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates and saves an,rocessed mesh consisting of primitive
    geometries.

    :param primitive_info: A list of dicts containing primitive params. Each dict must
        contain "name" which can for example be sphere, ellipsoid, box, etc. and
        "transform" which is a homogenous transformation matrix. The other params are
        primitive dependent but must be sufficient to construct that primitive.
    :param physical_properties: The physical properties.
    :param sdf_folder: The folder to write the sdf file to.
    :param is_hydroelastic: Whether to make the body rigid hydroelastic.
    :param prefix: An optional prefix for the processed mesh sdf file name.
    :param idx: An optional index for the processed mesh sdf file name.
    :param visual_mesh_file_path: The path to the mesh to use for the visual geometry.
    """
    com = physical_properties.center_of_mass
    procesed_mesh_sdf_str = f"""
        <?xml version="1.0"?>
        <sdf version="1.7">
            <model name="processed_manipuland_mesh">
                <link name="{manipuland_base_link_name}">
                    <inertial>
                        <inertia>
                            <ixx>{physical_properties.inertia[0,0]}</ixx>
                            <ixy>{physical_properties.inertia[0,1]}</ixy>
                            <ixz>{physical_properties.inertia[0,2]}</ixz>
                            <iyy>{physical_properties.inertia[1,1]}</iyy>
                            <iyz>{physical_properties.inertia[1,2]}</iyz>
                            <izz>{physical_properties.inertia[2,2]}</izz>
                        </inertia>
                        <mass>{physical_properties.mass}</mass>
                        <pose>{com[0]} {com[1]} {com[2]} 0 0 0</pose>
                    </inertial>
        """

    if visual_mesh_file_path is not None:
        procesed_mesh_sdf_str += f"""
                    <visual name="visual">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{visual_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
                    </visual>
            """

    # Add the primitives
    for i, info in enumerate(primitive_info):
        transform = info["transform"]
        translation = transform[:3, 3]
        rotation = R.from_matrix(transform[:3, :3]).as_euler("XYZ")

        if info["name"] == "ellipsoid":
            radii = info["radii"]
            geometry = f"""
                <ellipsoid>
                    <radii>{radii[0]} {radii[1]} {radii[2]}</radii>
                </ellipsoid>
            """
        elif info["name"] == "sphere":
            radius = info["radius"]
            geometry = f"""
                <sphere>
                    <radius>{radius}</radius>
                </sphere>
            """
        elif info["name"] == "box":
            size = info["size"]
            geometry = f"""
                <box>
                    <size>{size[0]} {size[1]} {size[2]}</size>
                </box>
            """
        elif info["name"] == "cylinder":
            height = info["height"]
            radius = info["radius"]
            geometry = f"""
                <cylinder>
                    <radius>{radius}</radius>
                    <length>{height}</length>
                </cylinder>
            """
        else:
            raise RuntimeError(f"Unsupported primitive type: {info['name']}")

        procesed_mesh_sdf_str += f"""
            <collision name="collision_{i}">
                <pose>
                    {translation[0]} {translation[1]} {translation[2]} {rotation[0]} {rotation[1]} {rotation[2]}
                </pose>
                <geometry>
                    {geometry}
                </geometry>
            """

        assert (
            not is_hydroelastic or physical_properties.mesh_resolution_hint is not None
        ), "Require a mesh resolution hint for Hydroelastic primitive collision geometries!"
        procesed_mesh_sdf_str += construct_drake_proximity_properties_sdf_str(
            physical_properties, is_hydroelastic
        )

        procesed_mesh_sdf_str += """
                </collision>
            """

    procesed_mesh_sdf_str += """
                    </link>
                </model>
            </sdf>
        """

    procesed_mesh_sdf_path = os.path.join(
        sdf_folder, f"{prefix}processed_mesh_{idx}.sdf"
    )
    with open(procesed_mesh_sdf_path, "w") as f:
        f.write(procesed_mesh_sdf_str)

    return procesed_mesh_sdf_path


def create_directive_str_for_sdf_path(
    sdf_path: str,
    model_name: str,
):
    directive_str = f"""
        directives:
        - add_model:
            name: {model_name}
            file: package://sim2sim/{sdf_path}
    """
    return directive_str


def create_processed_mesh_primitive_directive_str(
    primitive_info: List[Dict[str, Any]],
    physical_properties: PhysicalProperties,
    sdf_folder: str,
    model_name: str,
    manipuland_base_link_name: str,
    hydroelastic: str,
    prefix: str = "",
    idx: int = 0,
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates a directive for the processed mesh that contains primitive geometries.

    :param primitive_info: A list of dicts containing primitive params. Each dict must
        contain "name" which can for example be sphere, ellipsoid, box, etc. and
        "transform" which is a homogenous transformation matrix. The other params are
        primitive dependent but must be sufficient to construct that primitive.
    :param physical_properties: The physical properties.
    :param sdf_folder: The folder to write the sdf file to.
    :param model_name: The name of the directive model.
    :param hydroelastic: Whether to make the body rigid hydroelastic.
    :param prefix: An optional prefix for the processed mesh sdf file name.
    :param idx: An optional index for the processed mesh sdf file name.
    :param visual_mesh_file_path: The path to the mesh to use for the visual geometry.
    """
    procesed_mesh_sdf_path = create_processed_mesh_primitive_sdf_file(
        primitive_info,
        physical_properties,
        sdf_folder,
        manipuland_base_link_name,
        hydroelastic,
        prefix,
        idx,
        visual_mesh_file_path=visual_mesh_file_path,
    )
    processed_mesh_directive = f"""
        directives:
        - add_model:
            name: {model_name}
            file: package://sim2sim/{procesed_mesh_sdf_path}
    """
    return processed_mesh_directive


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


def get_hydroelastic_contact_viz_params() -> ContactVisualizerParams:
    """Contact visualizer params tuned for hydroelastic contact."""
    cparams = ContactVisualizerParams()
    cparams.force_threshold = 1e-2
    cparams.newtons_per_meter = 1e6
    cparams.newton_meters_per_meter = 1e1
    cparams.radius = 0.002
    return cparams


def get_point_contact_contact_viz_params() -> ContactVisualizerParams:
    """Contact visualizer params tuned for point contact."""
    cparams = ContactVisualizerParams()
    cparams.force_threshold = 1e-3
    cparams.newtons_per_meter = 1e2  # NOTE: Lower = larger force vectors
    cparams.newton_meters_per_meter = 1.0
    cparams.radius = 0.002
    return cparams


def copy_object_proximity_properties(
    const_proximity_properties: ProximityProperties,
    new_proximity_properties: ProximityProperties,
) -> None:
    """
    Copies properties from `const_proximity_properties` to `new_proximity_properties`.
    """
    for group_name in const_proximity_properties.GetGroupNames():
        properties = const_proximity_properties.GetPropertiesInGroup(group_name)
        for name in properties:
            if new_proximity_properties.HasProperty(group_name, name):
                continue
            new_proximity_properties.AddProperty(
                group_name,
                name,
                const_proximity_properties.GetProperty(group_name, name),
            )


def vector_pose_to_rigidtransform(pose: np.ndarray) -> RigidTransform:
    """Converts a pose of form [qw, qx, qy, qz, x, y, z] into a Drake RigidTransform."""
    quat = pose[:4]
    quat_normalized = quat / np.linalg.norm(quat)
    return RigidTransform(Quaternion(quat_normalized), pose[4:])


def get_principal_component(points: np.ndarray) -> np.ndarray:
    """
    :param points: Points of shape (N,K).
    :return: The principle component vector of shape (K,).
    """
    cov = np.cov(points.T)
    eigval, eigvec = np.linalg.eig(cov)
    order = eigval.argsort()
    principal_component = eigvec[:, order[-1]]
    return principal_component


def get_main_mesh_cluster(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Remove all but the largest mesh cluster."""
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    largest_cluster = copy.deepcopy(mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    largest_cluster.remove_triangles_by_mask(triangles_to_remove)

    return largest_cluster


def add_shape(
    plant: MultibodyPlant,
    shape: Shape,
    name: str,
    mass: float = 1.0,
    mu: float = 1.0,
    color: List[float] = [0.5, 0.5, 0.9, 1.0],
    add_contact_spheres: bool = False,
):
    """
    Adopted from https://github.com/RussTedrake/manipulation/blob/master/manipulation/scenarios.py.
    Adds a shape with inertia and friction to the plant.
    """
    instance = plant.AddModelInstance(name)
    if isinstance(shape, Box):
        inertia = UnitInertia.SolidBox(shape.width(), shape.depth(), shape.height())
    elif isinstance(shape, Cylinder):
        inertia = UnitInertia.SolidCylinder(shape.radius(), shape.length())
    elif isinstance(shape, Sphere):
        inertia = UnitInertia.SolidSphere(shape.radius())
    elif isinstance(shape, Capsule):
        inertia = UnitInertia.SolidCylinder(shape.radius(), shape.length())
    else:
        raise RuntimeError(f"need to write the unit inertia for shapes of type {shape}")
    body = plant.AddRigidBody(
        name,
        instance,
        SpatialInertia(mass=mass, p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=inertia),
    )
    if plant.geometry_source_is_registered():
        if isinstance(shape, Box):
            plant.RegisterCollisionGeometry(
                body,
                RigidTransform(),
                Box(
                    shape.width() - 0.001,
                    shape.depth() - 0.001,
                    shape.height() - 0.001,
                ),
                name,
                CoulombFriction(mu, mu),
            )
            if add_contact_spheres:
                i = 0
                for x in [-shape.width() / 2.0, shape.width() / 2.0]:
                    for y in [-shape.depth() / 2.0, shape.depth() / 2.0]:
                        for z in [-shape.height() / 2.0, shape.height() / 2.0]:
                            plant.RegisterCollisionGeometry(
                                body,
                                RigidTransform([x, y, z]),
                                Sphere(radius=1e-7),
                                f"contact_sphere{i}",
                                CoulombFriction(mu, mu),
                            )
                            i += 1
        else:
            plant.RegisterCollisionGeometry(
                body, RigidTransform(), shape, name, CoulombFriction(mu, mu)
            )

        plant.RegisterVisualGeometry(body, RigidTransform(), shape, name, color)

    return instance


def convert_obj_to_vtk(
    obj_meshes: List[o3d.geometry.TriangleMesh],
    output_path: str,
    tmp_folder_path: str = "convert_obj_to_vtk_tmp_folder",
) -> List[str]:
    """
    Converts OBJ file(s) to VTK files(s).

    :param obj_meshes: The meshes to convert.
    :param output_path: The path to write the VTK file or directory to write the VTK
        files to.
    :param tmp_folder_path: The temporary folder to store the intermediate results in.
        The folder will be created at the beginning and deleted at the end.
    :return: The VTK file paths.
    """
    os.mkdir(tmp_folder_path)

    def convert(path: str) -> str:
        name = Path(path).parts[-1][:-3]
        msh_path = os.path.join(tmp_folder_path, name + "msh")
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(
                ["FloatTetwild_bin", "-i", path, "-o", msh_path],
                stdout=devnull,
                stderr=subprocess.STDOUT,
            )
        mesh = meshio.read(msh_path)
        vtk_path = os.path.join(output_path, name + "vtk")
        mesh.write(vtk_path)
        return vtk_path

    vtk_file_paths = []
    if len(obj_meshes) > 1:
        assert os.path.isdir(
            output_path
        ), "The output path must be a directory if the input is multiple meshes!"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for i, obj_mesh in tqdm(
            enumerate(obj_meshes), total=len(obj_meshes), desc="Converting OBJ to VTK"
        ):
            obj_path = os.path.join(tmp_folder_path, f"mesh_piece_{i:03d}.obj")
            o3d.io.write_triangle_mesh(obj_path, obj_mesh)
            vtk_path = convert(obj_path)
            vtk_file_paths.append(vtk_path)
    else:
        assert output_path[-3:].lower() == "vtk"
        obj_path = os.path.join(tmp_folder_path, "mesh.obj")
        o3d.io.write_triangle_mesh(obj_path, obj_meshes[0])
        vtk_path = convert(obj_path)
        vtk_file_paths.append(vtk_path)

    shutil.rmtree(tmp_folder_path)

    return vtk_file_paths
