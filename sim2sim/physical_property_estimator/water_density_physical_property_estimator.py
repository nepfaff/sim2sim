import open3d as o3d

from sim2sim.util import calc_mesh_inertia, PhysicalProperties
from .physical_property_estimator_base import PhysicalPropertyEstimatorBase


class WaterDensityPhysicalPropertyEstimator(PhysicalPropertyEstimatorBase):
    """
    Estimates mass and inertia by assuming a constant density of water.
    """

    def __init__(self):
        super().__init__()

    def estimate_physical_properties(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> PhysicalProperties:
        """
        :param mesh: The mesh to estimate physcial properties for.
        :return: The estimated physical properties.
        """
        mass, inertia, com = calc_mesh_inertia(mesh)
        # TODO: Also estimate compliant Hydroelastic properties
        properties = PhysicalProperties(
            mass=mass, inertia=inertia, center_of_mass=com, is_compliant=False
        )
        return properties
