import os
from typing import List, Dict, Any, Optional
import copy

import open3d as o3d
import trimesh
import numpy as np
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


def create_processed_mesh_sdf_file(
    physical_properties: PhysicalProperties,
    processed_mesh_file_path: str,
    tmp_folder: str,
    manipuland_base_link_name: str,
    hydroelastic: bool,
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates and saves an SDF file for the processed mesh.

    :param physical_properties: The physical properties.
    :param processed_mesh_file_path: The path to the processed mesh obj file.
    :param tmp_folder: The folder to write the sdf file to.
    :param hydroelastic: Whether to make the body rigid hydroelastic.
    :return procesed_mesh_sdf_path: The path to the SDF file.
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

    procesed_mesh_sdf_str += f"""
                    <collision name="collision">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{processed_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
        """

    if hydroelastic:
        procesed_mesh_sdf_str += """
                <drake:proximity_properties>
                    <drake:rigid_hydroelastic/>
                </drake:proximity_properties>
            """

    procesed_mesh_sdf_str += """
                    </collision>
                    </link>
                </model>
            </sdf>
        """

    idx = processed_mesh_file_path.find("sim2sim/")
    procesed_mesh_sdf_path = processed_mesh_file_path[idx + 8 :].replace(".obj", ".sdf")

    with open(procesed_mesh_sdf_path, "w") as f:
        f.write(procesed_mesh_sdf_str)
    return procesed_mesh_sdf_path


def create_decomposition_processed_mesh_sdf_file(
    physical_properties: PhysicalProperties,
    processed_mesh_file_path: str,
    mesh_pieces: List[str],
    sdf_folder: str,
    manipuland_base_link_name: str,
    hydroelastic: bool,
    prefix: str = "",
    parts_are_convex: bool = True,
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates and saves an SDF file for the processed decomposition of a mesh.

    :param physical_properties: The physical properties.
    :param processed_mesh_file_path: The path to the processed mesh obj file.
    :param mesh_pieces: A list of mesh piece paths.
    :param sdf_folder: The folder to write the sdf file to.
    :param hydroelastic: Whether to make the body rigid hydroelastic.
    :param prefix: An optional prefix for the processed mesh sdf file name.
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

        if hydroelastic:
            procesed_mesh_sdf_str += """
                    <drake:proximity_properties>
                        <drake:rigid_hydroelastic/>
                    </drake:proximity_properties>
                """

        procesed_mesh_sdf_str += """
                </collision>
            """

    procesed_mesh_sdf_str += """
                    </link>
                </model>
            </sdf>
        """

    procesed_mesh_sdf_path = os.path.join(sdf_folder, f"{prefix}processed_mesh.sdf")
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
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates a directive for the processed mesh.

    :param physical_properties: The physical properties.
    :param processed_mesh_file_path: The path to the processed mesh obj file.
    :param sdf_folder: The folder to write the sdf file to.
    :param model_name: The name of the directive model.
    :param hydroelastic: Whether to make the body rigid hydroelastic.
    :param prefix: An optional prefix for the processed mesh sdf file name.
    :param visual_mesh_file_path: The path to the mesh to use for the visual geometry.
    """
    if not (processed_mesh_file_path).endswith(".obj"):
        listed_files = [
            os.path.join(processed_mesh_file_path, f)
            for f in os.listdir(processed_mesh_file_path)
            if "mesh_piece_" in f
        ]
        procesed_mesh_sdf_path = create_decomposition_processed_mesh_sdf_file(
            physical_properties,
            processed_mesh_file_path + ".obj",
            listed_files,
            sdf_folder,
            manipuland_base_link_name,
            hydroelastic,
            prefix,
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
    hydroelastic: bool,
    prefix: str = "",
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates and saves an SDF file for a processed mesh consisting of primitive geometries.

    :param primitive_info: A list of dicts containing primitive params. Each dict must contain "name" which can for
        example be sphere, ellipsoid, box, etc. and "transform" which is a homogenous transformation matrix. The other
        params are primitive dependent but must be sufficient to construct that primitive.
    :param physical_properties: The physical properties.
    :param sdf_folder: The folder to write the sdf file to.
    :param hydroelastic: Whether to make the body rigid hydroelastic.
    :param prefix: An optional prefix for the processed mesh sdf file name.
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
                <pose>{translation[0]} {translation[1]} {translation[2]} {rotation[0]} {rotation[1]} {rotation[2]}</pose>
                <geometry>
                    {geometry}
                </geometry>
            """

        if hydroelastic:
            procesed_mesh_sdf_str += """
                    <drake:proximity_properties>
                        <drake:rigid_hydroelastic/>
                        <drake:mesh_resolution_hint>0.01</drake:mesh_resolution_hint>
                    </drake:proximity_properties>
                """

        procesed_mesh_sdf_str += """
                </collision>
            """

    procesed_mesh_sdf_str += """
                    </link>
                </model>
            </sdf>
        """

    procesed_mesh_sdf_path = os.path.join(sdf_folder, f"{prefix}processed_mesh.sdf")
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
    visual_mesh_file_path: Optional[str] = None,
) -> str:
    """
    Creates a directive for the processed mesh that contains primitive geometries.

    :param primitive_info: A list of dicts containing primitive params. Each dict must contain "name" which can for
        example be sphere, ellipsoid, box, etc. The other params are primitive dependent but must be sufficient to
        construct that primitive.
    :param physical_properties: The physical properties.
    :param sdf_folder: The folder to write the sdf file to.
    :param model_name: The name of the directive model.
    :param hydroelastic: Whether to make the body rigid hydroelastic.
    :param prefix: An optional prefix for the processed mesh sdf file name.
    :param visual_mesh_file_path: The path to the mesh to use for the visual geometry.
    """
    procesed_mesh_sdf_path = create_processed_mesh_primitive_sdf_file(
        primitive_info,
        physical_properties,
        sdf_folder,
        manipuland_base_link_name,
        hydroelastic,
        prefix,
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
    """Copies properties from `const_proximity_properties` to `new_proximity_properties`."""
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
