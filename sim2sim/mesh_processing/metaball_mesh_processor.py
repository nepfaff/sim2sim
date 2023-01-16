from typing import Dict, Any

import open3d as o3d
import numpy as np
import trimesh
import sklearn.mixture

from .mesh_processor_base import MeshProcessorBase
from sim2sim.util import open3d_to_trimesh
from sim2sim.logging import DynamicLoggerBase


class MetaBallMeshProcessor(MeshProcessorBase):
    """Replaces the mesh with spheres obtained from fitting GMMs using Expectation Maximization."""

    def __init__(self, logger: DynamicLoggerBase, visualize: bool, gmm_em_params: Dict[str, Any], threshold_std: float):
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

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        :param mesh: The mesh.
        :return: The simplified mesh.
        """
        tmesh = open3d_to_trimesh(mesh)
        pts = trimesh.sample.sample_surface_even(tmesh, 10000)[0]
        gmm = sklearn.mixture.GaussianMixture(**self._gmm_em_params)
        gmm.fit(pts)
        mean = gmm.means_
        prec = gmm.precisions_cholesky_
        covariance = np.linalg.inv(prec)

        # use sphere to approximate this
        max_radius = covariance.reshape(-1, 9).max(-1) * self._threshold_std
        centers = []
        radius = []
        output_meshes = []

        for idx in range(self._gmm_em_params["n_components"]):
            c_i, r_i = mean[idx], max_radius[idx]
            centers.append(c_i)
            radius.append(r_i)
            sphere = o3d.geometry.TriangleMesh.create_sphere(r_i)
            sphere.paint_uniform_color((0.0, 0.0, 0.8))
            sphere.translate(c_i)
            output_meshes.append(sphere)

        if self._visualize:
            viewer = o3d.visualization.Visualizer()
            viewer.create_window()
            pcd = mesh.sample_points_uniformly(number_of_points=50000)
            viewer.add_geometry(pcd)
            for sphere in output_meshes:
                viewer.add_geometry(sphere)

            opt = viewer.get_render_option()
            opt.show_coordinate_frame = True
            viewer.run()
            viewer.destroy_window()

        return None, output_meshes
