import open3d as o3d
import numpy as np
import torch
import trimesh
import pointnet2_ops.pointnet2_utils as pointnet2_utils

from sim2sim.util import open3d_to_trimesh
from sim2sim.logging import DynamicLoggerBase
from .mesh_processor_base import MeshProcessorBase


class SphereMeshProcessor(MeshProcessorBase):
    """Replaces a mesh with spheres using farthest point sampling."""

    def __init__(self, logger: DynamicLoggerBase, target_sphere_num: int, visualize: bool):
        """
        :param target_sphere_num: The number of spheres that the simplified mesh should contain.
        :param visualize: Whether to visualize the fitted spheres.
        """
        super().__init__(logger)

        self._target_sphere_num = target_sphere_num
        self._visualize = visualize

    def process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        :param mesh: The mesh.
        :return: The simplified mesh mesh.
        """
        tmesh = open3d_to_trimesh(mesh)
        points = np.array(trimesh.sample.sample_surface_even(tmesh, 10000)[0])
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

            c_i, r_i, _ = trimesh.nsphere.fit_nsphere(within_sphere_points)
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
