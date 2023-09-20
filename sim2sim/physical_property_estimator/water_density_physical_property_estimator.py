from typing import List

import open3d as o3d

from sim2sim.util import PhysicalProperties, calc_mesh_inertia

from .physical_property_estimator_base import PhysicalPropertyEstimatorBase


class WaterDensityPhysicalPropertyEstimator(PhysicalPropertyEstimatorBase):
    """
    Estimates mass and inertia by assuming a constant density of water.
    """

    def __init__(self):
        super().__init__()

    def estimate_physical_properties(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[PhysicalProperties]:
        """
        :param meshes: The meshes to estimate physcial properties for.
        :return: The estimated physical properties.
        """
        properties = []
        for mesh in meshes:
            mass, inertia, com = calc_mesh_inertia(mesh)
            # TODO: Also estimate compliant Hydroelastic properties
            properties.append(
                PhysicalProperties(
                    mass=mass, inertia=inertia, center_of_mass=com, is_compliant=False
                )
            )
        return properties
