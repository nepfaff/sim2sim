from typing import List, Optional

import open3d as o3d
import numpy as np

from .physical_property_estimator_base import PhysicalPropertyEstimatorBase
from sim2sim.util import PhysicalProperties


class GTPhysicalPropertyEstimator(PhysicalPropertyEstimatorBase):
    """
    Returns the given physical properties.
    """

    def __init__(
        self,
        mass: float,
        inertia: List[List[float]],
        center_of_mass: List[float],
        is_compliant: bool,
        hydroelastic_modulus: Optional[float] = None,
        hunt_crossley_dissipation: Optional[float] = None,
        mu_dynamic: Optional[float] = None,
        mu_static: Optional[float] = None,
        mesh_resolution_hint: Optional[float] = None,
    ):
        """
        :param mass: The mass in kg.
        :param inertia: The moment of inertia of shape (3,3).
        :param center_of_mass: The center of mass that the inertia is about of shape (3,).
        :param is_compliant: Whether the object is compliant or rigid in
            case of a Hydroelastic contact model. If compliant, the compliant
            Hydroelastic arguments are required.
        :param hydroelastic_modulus: This is the measure of how stiff the material is.
            It directly defines how much pressure is exerted given a certain amount of
            penetration. More pressure leads to greater forces. Larger values create
            stiffer objects.
        :param hunt_crossley_dissipation: A non-negative real value. This gives the
            contact an enerDynamic coefficient of friction.
        :param mu_static: Static coefficient of friction. Not used in discrete systems.
        :param mesh_resolution_hint: A positive real value in meters. Most shapes
            (capsules, cylinders, ellipsoids, spheres) need to be tessellated into
            meshes. The resolution hint controls the fineness of the meshes. It is a
            no-op for mesh geometries and is consequently not required for mesh
            geometries.
        """
        super().__init__()

        if is_compliant:
            assert (
                hydroelastic_modulus is not None
            ), "Compliant Hydroelastic objects require a Hydroelastic modulus!"

        self._physical_properties = PhysicalProperties(
            mass=mass,
            inertia=np.asarray(inertia),
            center_of_mass=np.asarray(center_of_mass),
            is_compliant=is_compliant,
            hydroelastic_modulus=hydroelastic_modulus,
            hunt_crossley_dissipation=hunt_crossley_dissipation,
            mu_dynamic=mu_dynamic,
            mu_static=mu_static,
            mesh_resolution_hint=mesh_resolution_hint,
        )

    def estimate_physical_properties(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> PhysicalProperties:
        """
        :param mesh: The mesh to estimate physcial properties for.
        :return: The estimated physical properties.
        """
        return self._physical_properties
