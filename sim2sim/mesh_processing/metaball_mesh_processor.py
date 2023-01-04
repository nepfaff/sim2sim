import open3d as o3d

from .mesh_processor_base import MeshProcessorBase
import pytorch3d
import pytorch3d.ops as ops
import numpy as np
import torch
import IPython
import trimesh
import scipy.spatial
from trimesh import util, convex

import sklearn.mixture
from .utils import open3d_to_trimesh


class MetaBallMeshProcessor(MeshProcessorBase):
    """Implements mesh processing through quadric decimation."""

    def __init__(self, target_sphere_num: int):
        """
        :param target_sphere_num: The number of spheres that the simplified mesh should contain.
        """
        super().__init__()

        # change this to spit out 100 meshes
        self._target_sphere_num = target_sphere_num

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        :param mesh: The mesh.
        :return: The simplified mesh mesh.
        """
        vis = True
        std = 1
        NUM_MIXTURE = self._target_sphere_num
        tmesh = open3d_to_trimesh(mesh)
        pts = trimesh.sample.sample_surface_even(tmesh, 10000)[0]
        gmm = sklearn.mixture.GaussianMixture(NUM_MIXTURE)
        gmm.fit(pts)
        weights = gmm.weights_
        weight_log = np.log(weights)
        mean = gmm.means_
        prec = gmm.precisions_cholesky_
        covariance = np.linalg.inv(prec)

        # use sphere to approximate this
        max_radius = covariance.reshape(-1, 9).max(-1) * std

        # IPython.embed()
        # limit the number of points but keep max radius
        centers = []
        radius = []
        output_meshes = []

        for idx in range(self._target_sphere_num):
            c_i, r_i = mean[idx], max_radius[idx]
            centers.append(c_i)
            radius.append(r_i)
            sphere = o3d.geometry.TriangleMesh.create_sphere(r_i)
            sphere.paint_uniform_color((0.0, 0.0, 0.8))
            sphere.translate(c_i)
            output_meshes.append(sphere)

        if vis:
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

        # TODO(liruiw): convert this to a list of sphere geometry?
        # return [centers, radius]
        return None, output_meshes

        # create a voronoi region
        # simplified_mesh = mesh.simplify_quadric_decimation(1000)
        # voronoi = scipy.spatial.Voronoi(points, furthest_site=True)
        # radii_2 = scipy.spatial.distance.cdist(
        #     voronoi.vertices, points,
        #     metric='sqeuclidean').max(axis=1)
        # npoint = torch.cuda.FloatTensor([self._target_sphere_num])
        # trimesh.nsphere.fit_nsphere()
        # radius_v = np.sqrt(radii_2[radii_idx]) * points_scale
        # center_v = (voronoi.vertices[radii_idx] *
        #         points_scale) + points_origin

        # new_xyz, farthest_pt_idx = ops.sample_farthest_points(xyz[None], npoint)

        # using a surrogate for maximum distance

        # pick some radius for querying
        # radius_lo = 0.2
        # radius_hi
        # radius = torch.rand(self._target_sphere_num).uniform_(radius_lo, radius_hi).cuda()
