from inverse_graphics_base import InverseGraphicsBase
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


class BasicInverseGraph(InverseGraphicsBase):
    """
    Basic Inverse Graph generator with Poisson Surface

    Args:
        images (numpy array): The array of rgb values of the images
        intrinsics (numpy array): The intrinsics values of the camera
        extrinsics (numpy array): The extrinsics values of the camera
        depth (numpy array, optional): The depth values of the images

    Attributes:
        images (numpy array): The array of rgb values of the images
        intrinsics (numpy array): The intrinsics values of the camera
        extrinsics (numpy array): The extrinsics values of the camera
        depth (numpy array): The depth values of the images
    """

    def __init__(self, images, intrinsics, extrinsics, depth=None):
        super().__init__(images, intrinsics, extrinsics, depth)
        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.depth = depth

    def checkMaskMax(self, arr) -> np.ndarray:
        """
        Check for the max value in the binary mask array and change it to -1.
        Open3d disregards values of <0 when generating PointClouds.

        Args:
            arr (numpy array): The array of the mask image.

        Returns:
            numpy array with mask values of -1
        """

        if np.max(arr) > 0:
            newArr = np.where(arr > 0, -1, arr)
        else:
            return arr
        return newArr

    def applyMask(self, image, mask) -> np.ndarray:
        """
        Apply masked values into images

        Args:
            image (numpy array): The array of the RGB or Depth image.
            mask (numpy array): The array of the mask image.

        Returns:
            new image numpy array
        """

        newMask = self.checkMaskMax(mask)
        npImage = np.asarray(image)
        maskImg = npImage.copy()
        maskImg = npImage * newMask
        return o3d.geometry.Image(maskImg.astype(np.float32))

    def convertImage2RGBD(self, images=None, depth=None) -> o3d.geometry.RGBDImage:
        """
        Converts RGB images into RGBD images with depth matrix

        Args:
            images (numpy array): The array of the RGB image.
            depth (numpy array): The array of the depth image.

        Returns:
            open3d RGDBD image.
        """
        if images is not None and depth is not None:
            images = images.astype(np.float32)
            depth = depth.astype(np.float32)
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(images),
                o3d.geometry.Image(depth),
            )

        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(self.images.astype(np.float32)),
            o3d.geometry.Image(self.depth.astype(np.float32)),
        )

    def convertImage2PC(self) -> o3d.geometry.PointCloud:
        """
        Converts RGBD or Depth images into PointClouds. Takes into account camera intrinsics and extrinsics

        Returns:
            open3d PointCloud.
        """
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

    def convertImage2PCWMask(self, mask) -> o3d.geometry.PointCloud:
        """
        Converts RGBD or Depth and mask images into PointClouds. Takes into account camera intrinsics and extrinsics

        Args:
            mask (numpy array): The array of the mask image.

        Returns:
            open3d PointCloud.
        """
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self._images)
        # self.pointClouds = pcd.points

        input_image = self.convertImage2RGBD(self.images, self.depth)
        # input_image = self.convertImage2RGBD(
        #     self.applyMask(self.images, mask), self.applyMask(self.depth, mask)
        # )
        print(f"input_image={input_image}")
        # input_image = self.applyMask(input_image, mask)
        print(input_image.color)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=input_image,
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=np.asarray(input_image.color).shape[0],
                height=np.asarray(input_image.color).shape[1],
                intrinsic_matrix=self.intrinsics,
            ),
            extrinsic=self.extrinsics,
        )

        # if self.depth is None:
        #     input_image = self.convertImage2RGBD(
        #         self.applyMask(self.images, mask), self.applyMask(self.depth, mask)
        #     )
        #     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #         image=input_image,
        #         intrinsic=o3d.camera.PinholeCameraIntrinsic(
        #             width=input_image.shape[0],
        #             height=input_image.shape[1],
        #             intrinsic_matrix=self.intrinsics,
        #         ),
        #         extrinsic=self.extrinsics,
        #     )
        # else:
        #     input_image = self.applyMask(self.depth, mask)
        #     pcd = o3d.geometry.PointCloud.create_from_depth_image(
        #         image=input_image,
        #         intrinsic=o3d.camera.PinholeCameraIntrinsic(
        #             width=input_image.shape[0],
        #             height=input_image.shape[1],
        #             intrinsic_matrix=self.intrinsics,
        #         ),
        #         extrinsic=self.extrinsics,
        #     )

        pcd = pcd.transform(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )  # Flip it, otherwise the pointcloud will be upside down

        return pcd

    def run(self, pointCloud=None) -> Tuple[o3d.geometry.TriangleMesh, o3d.utility.DoubleVector]:
        """
        Converts PointClouds into mesh and density with Poisson surface recontruction

        Returns:
            Number of triangle mesh and the density.
        """
        print("run Poisson surface reconstruction")
        if pointCloud is None:
            pcd = self.convertImage2PC()
        else:
            pcd = pointCloud

        # pcd.estimate_normals()
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        # exit()
        pcd.estimate_normals()
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        print(mesh)
        return mesh, densities


def main():
    sample_path = "../../data/sample_data/"
    image_path = sample_path + "images/image0000.png"
    depth_path = sample_path + "depths/depth0000.txt"
    mask_path = sample_path + "binary_masks/mask0000.png"
    intrinsic_path = sample_path + "intrinsics/intrinsics0000.txt"
    extrinsic_path = sample_path + "camera_poses/pose0000.txt"
    # image_path = "image0000.png"
    colourRaw = np.asarray(o3d.io.read_image(image_path))
    depthRaw = np.loadtxt(depth_path)
    maskImg = np.asarray(o3d.io.read_image(mask_path))
    intrinsic = np.loadtxt(intrinsic_path)
    extrinsic = np.loadtxt(extrinsic_path)

    # print(colourRaw.shape)
    # print(colourRaw.shape[1])
    # exit()

    basicGraph = BasicInverseGraph(colourRaw, intrinsic, extrinsic, depthRaw)
    pcd = basicGraph.convertImage2PCWMask(maskImg)
    mesh, densities = basicGraph.run(pcd)
    # print(mesh)
    # print(densities)
    # print(colourRaw)
    # print(depthRaw)
    # print(image_path)


if __name__ == "__main__":
    main()
