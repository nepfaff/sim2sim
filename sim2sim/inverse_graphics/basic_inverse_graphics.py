from inverse_graphics_base import InverseGraphicsBase
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


class BasicInverseGraph(InverseGraphicsBase):

    pointClouds = None

    def __init__(self, images, intrinsics, extrinsics, depth=None):
        """TODO"""
        super().__init__(images, intrinsics, extrinsics, depth)
        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.depth = depth

    def checkMinMax(self, arr):
        if np.max(arr) > 0:
            newArr = np.where(arr > 0, -1, arr)
        else:
            return arr
        return newArr

    def applyMask(self, mask):
        newMask = self.checkMinMax(mask)
        maskImg = self.images.copy()
        maskImg[:, :] = self.images.img[:, :] * newMask[:, :]
        return maskImg

    def convertImage2RGBD(self):
        """TODO"""
        return o3d.geometry.RGBDImage.create_from_color_and_depth(self.images, self.depth)

    def convertImage2PC(self):
        """TODO"""
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self._images)
        # self.pointClouds = pcd.points

        if self.depth is None:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                image=self.convertImage2RGBD(),
                intrinsic=o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=self.intrinsics),
                extrinsic=self.extrinsics,
            )
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                image=self.depth,
                intrinsic=o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=self.intrinsics),
                extrinsic=self.extrinsics,
            )
        pcd = pcd.transform(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )  # Flip it, otherwise the pointcloud will be upside down

        return pcd

    def convertImage2PCWMask(self, mask):
        """TODO"""
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self._images)
        # self.pointClouds = pcd.points

        if self.depth is None:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                image=self.convertImage2RGBD(),
                intrinsic=o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=self.intrinsics),
                extrinsic=self.extrinsics,
            )
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                image=self.depth,
                intrinsic=o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=self.intrinsics),
                extrinsic=self.extrinsics,
            )
        pcd = pcd.transform(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )  # Flip it, otherwise the pointcloud will be upside down

        return pcd

    def run(self):
        """TODO"""
        print("run Poisson surface reconstruction")
        pcd = self.convertImage2PC()
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        print(mesh)
