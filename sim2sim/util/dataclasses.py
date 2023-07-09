from dataclasses import dataclass
from typing import Union, List, Dict, Any
from enum import Enum

import numpy as np
import open3d as o3d


@dataclass
class PhysicalProperties:
    """
    A dataclass for physical properties.

    See https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for
    more information about these properties.
    """

    mass: float
    """Mass in kg."""
    inertia: np.ndarray
    """Moment of inertia of shape (3,3)."""
    center_of_mass: np.ndarray
    """The center of mass that the inertia is about of shape (3,)."""
    is_compliant: bool
    """Whether the object is compliant or rigid in case of a Hydroelastic contact model.
    If compliant, the compliant Hydroelastic arguments are required."""
    hydroelastic_modulus: Union[float, None] = None
    """This is the measure of how stiff the material is. It directly defines how much
    pressure is exerted given a certain amount of penetration. More pressure leads to
    greater forces. Larger values create stiffer objects."""
    hunt_crossley_dissipation: Union[float, None] = None
    """A non-negative real value. This gives the contact an energy-damping property."""
    mu_dynamic: Union[float, None] = None
    """Dynamic coefficient of friction."""
    mu_static: Union[float, None] = None
    """Static coefficient of friction. Not used in discrete systems."""
    mesh_resolution_hint: Union[float, None] = None
    """A positive real value in meters. Most shapes (capsules, cylinders, ellipsoids,
    spheres) need to be tessellated into meshes. The resolution hint controls the
    fineness of the meshes. It is a no-op for mesh geometries and is consequently not
    required for mesh geometries."""


@dataclass
class MeshProcessorResult:
    """
    A dataclass for storing a mesh processor result.

    Only one of many options is stored, as determined by `result_type`. This result
    should be accessed with `get_result()` rather than reading the class attributes
    directly.
    """

    class ResultType(Enum):
        TRIANGLE_MESH = 1
        PRIMITIVE_INFO = 2
        SDF_PATH = 3
        VTK_PATHS = 4

    result_type: ResultType
    """The type of mesh processor result that this object contains."""
    triangle_meshes: Union[List[o3d.geometry.TriangleMesh], None] = None
    """A list of mesh parts that form the processed mesh."""
    primitive_info: Union[List[Dict[str, Any]], None] = None
    """A list of dicts containing primitive params. Each dict must contain "name" which
    can for example be sphere, ellipsoid, box, etc. and "transform" which is a
    homogenous transformation matrix. The other params are primitive dependent but must
    be sufficient to construct that primitive."""
    sdf_path: Union[str, None] = None
    """A path to an SDFormat file to use directly instead of constructing it from the
    pipeline results."""
    vtk_paths: Union[List[str], None] = None
    """A list of vtk file paths. The combined vtk objects form the processed mesh."""

    def get_result(self):
        """Returns the stored mesh processor result."""
        if self.result_type == self.ResultType.TRIANGLE_MESH:
            return self.triangle_meshes
        if self.result_type == self.ResultType.PRIMITIVE_INFO:
            return self.primitive_info
        if self.result_type == self.ResultType.SDF_PATH:
            return self.sdf_path
        if self.result_type == self.ResultType.VTK_PATHS:
            return self.vtk_paths
