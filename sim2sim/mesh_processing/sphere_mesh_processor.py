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
import pointnet2_ops.pointnet2_utils as pointnet2_utils


class SphereMeshProcessor(MeshProcessorBase):
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
        points = np.array(mesh.vertices)
        print(
            points[:, 0].max(),
            points[:, 0].min(),
            points[:, 1].max(),
            points[:, 1].min(),
            points[:, 2].max(),
            points[:, 2].min(),
        )
        points = torch.from_numpy(points).cuda().float().contiguous()[None]

        # subsampled_pts = pointnet2_utils.furthest_point_sample(points, self._target_sphere_num)
        subsampled_pts = pointnet2_utils.gather_operation(
            points.transpose(1, 2).contiguous(),
            pointnet2_utils.furthest_point_sample(points[..., :3].contiguous(), self._target_sphere_num),
        ).contiguous()
        avg_point_num = points.shape[1] // self._target_sphere_num

        # limit the number of points but keep max radius
        centers = []
        radius = []
        output_meshes = []
        remaining_points = points.clone()

        for idx in range(self._target_sphere_num):
            dist = subsampled_pts[..., [idx]] - torch.cat(
                (subsampled_pts[..., :idx], subsampled_pts[..., idx + 1 :]), dim=-1
            )
            dist = dist.norm(dim=1).min()
            output = pointnet2_utils.ball_query(
                dist,
                avg_point_num,
                remaining_points.contiguous(),
                subsampled_pts[..., [idx]].transpose(1, 2).float().contiguous(),
            )
            output = output.detach().cpu().numpy()[0][0]
            within_sphere_points = remaining_points[0][output].detach().cpu().numpy()

            # remove these points
            index = torch.ones(remaining_points.shape[1], dtype=bool).cuda()
            index[output] = False
            remaining_points = remaining_points[:, index]

            c_i, r_i, err = trimesh.nsphere.fit_nsphere(within_sphere_points)
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
