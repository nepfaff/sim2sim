from typing import List, Any, Dict

import open3d as o3d
import numpy as np
import trimesh
import sklearn.mixture

try:
    from learning_real2sim.src.ellipsoid import Ellipsoid
except:
    print("Failed to import learning_real2sim. GMMMeshProcessor won't be working!")

from .mesh_processor_base import MeshProcessorBase
from sim2sim.util import open3d_to_trimesh, MeshProcessorResult
from sim2sim.logging import DynamicLogger

DIV_EPSILON = 1e-9


class GMMMeshProcessor(MeshProcessorBase):
    """Replaces the mesh with spheres obtained from fitting GMMs using Expectation Maximization."""

    def __init__(
        self,
        logger: DynamicLogger,
        visualize: bool,
        gmm_em_params: Dict[str, Any],
        threshold_std: float,
    ):
        """
        :param logger: The logger.
        :param visualize: Whether to visualize the fitted spheres.
        :param gmm_em_params: Parameters for GMM EM fitting. All must be valid arguments for
            sklearn.mixture.GaussianMixture.
        :param threshold_std: The standard deviation to use as a threshold for converting a GMM into a mesh.
        """
        super().__init__(logger)

        self._visualize = visualize
        self._gmm_em_params = gmm_em_params
        self._threshold_std = threshold_std

        self._logger.log(meta_data={"mesh_processing_GMM_EM": gmm_em_params})

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> MeshProcessorResult:
        tmesh = open3d_to_trimesh(mesh)
        pts = trimesh.sample.sample_surface_even(tmesh, 10000)[0]
        gmm = sklearn.mixture.GaussianMixture(**self._gmm_em_params)
        gmm.fit(pts)
        mean = gmm.means_
        prec = gmm.precisions_
        rotation_matrix, radii_inv, _ = np.linalg.svd(prec)
        radii = np.sqrt(1.0 / (radii_inv + DIV_EPSILON))

        centers: List[np.ndarray] = []
        radius: List[np.ndarray] = []
        ellipsoids: List[Ellipsoid] = []

        for idx in range(self._gmm_em_params["n_components"]):
            c_i, r_i = mean[idx], radii[idx]
            centers.append(c_i)
            radius.append(r_i)
            rot_mat = np.eye(4)
            rot_mat[:3, :3] = rotation_matrix[idx]
            ellipsoid = Ellipsoid(
                center=c_i, radius=r_i * self._threshold_std, scale=1, transform=rot_mat
            )
            ellipsoids.append(ellipsoid)

        if self._visualize:
            scene = trimesh.scene.scene.Scene()
            for ellipsoid in ellipsoids:
                scene.add_geometry(ellipsoid)
            open3d_pcd = mesh.sample_points_uniformly(number_of_points=10000)
            trimesh_pcd = trimesh.points.PointCloud(
                np.asarray(open3d_pcd.points),
                colors=np.repeat([[0, 0, 150]], len(open3d_pcd.points), axis=0),
            )
            scene.add_geometry(trimesh_pcd)
            scene.show()

        analytical_ellipsoids = []
        for ellipsoid in ellipsoids:
            analytical_ellipsoids.append(
                {
                    "name": "ellipsoid",
                    "radii": np.asarray(ellipsoid.radius),
                    "transform": np.asarray(ellipsoid.transform),
                }
            )

        return MeshProcessorResult(
            result_type=MeshProcessorResult.ResultType.PRIMITIVE_INFO,
            primitive_info=analytical_ellipsoids,
        )
