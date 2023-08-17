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
        masses: List[float],
        inertias: List[List[List[float]]],
        centers_of_mass: List[List[float]],
        is_compliant: List[bool],
        hydroelastic_moduli: Optional[List[float]] = None,
        hunt_crossley_dissipations: Optional[List[float]] = None,
        mus_dynamic: Optional[List[float]] = None,
        mus_static: Optional[List[float]] = None,
        mesh_resolution_hints: Optional[List[float]] = None,
    ):
        """
        :param masses: The masses in kg.
        :param inertias: The moments of inertia of shape (N, 3,3).
        :param centers_of_mass: The centers of mass that the inertias are about of shape
            (N,3).
        :param is_compliant: Whether the object is compliant or rigid in
            case of a Hydroelastic contact model. If compliant, the compliant
            Hydroelastic arguments are required.
        :param hydroelastic_moduli: This is the measure of how stiff the material is.
            It directly defines how much pressure is exerted given a certain amount of
            penetration. More pressure leads to greater forces. Larger values create
            stiffer objects.
        :param hunt_crossley_dissipations: A non-negative real value. This gives the
            contact an enerDynamic coefficient of friction.
        :param mus_dynamic: Dynamic coefficient of friction.
        :param mus_static: Static coefficient of friction. Not used in discrete systems.
        :param mesh_resolution_hints: A positive real value in meters. Most shapes
            (capsules, cylinders, ellipsoids, spheres) need to be tessellated into
            meshes. The resolution hint controls the fineness of the meshes. It is a
            no-op for mesh geometries and is consequently not required for mesh
            geometries.
        """
        super().__init__()

        self._physical_properties = []
        length = len(masses)
        for (
            mass,
            inertia,
            center_of_mass,
            compliant,
            hydroelastic_modulus,
            hunt_crossley_dissipation,
            mu_dynamic,
            mu_static,
            mesh_resolution_hint,
        ) in zip(
            masses,
            inertias,
            centers_of_mass,
            is_compliant,
            [None] * length if hydroelastic_moduli is None else hydroelastic_moduli,
            (
                [None] * length
                if hunt_crossley_dissipations is None
                else hunt_crossley_dissipations
            ),
            [None] * length if mus_dynamic is None else mus_dynamic,
            [None] * length if mus_static is None else mus_static,
            [None] * length if mesh_resolution_hints is None else mesh_resolution_hint,
        ):
            if compliant:
                assert (
                    hydroelastic_modulus is not None
                ), "Compliant Hydroelastic objects require a Hydroelastic modulus!"

            self._physical_properties.append(
                PhysicalProperties(
                    mass=mass,
                    inertia=np.asarray(inertia),
                    center_of_mass=np.asarray(center_of_mass),
                    is_compliant=compliant,
                    hydroelastic_modulus=hydroelastic_modulus,
                    hunt_crossley_dissipation=hunt_crossley_dissipation,
                    mu_dynamic=mu_dynamic,
                    mu_static=mu_static,
                    mesh_resolution_hint=mesh_resolution_hint,
                )
            )

    def estimate_physical_properties(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[PhysicalProperties]:
        """
        :param meshes: The meshes to estimate physcial properties for.
        :return: The estimated physical properties.
        """
        assert len(meshes) == len(self._physical_properties)
        return self._physical_properties
