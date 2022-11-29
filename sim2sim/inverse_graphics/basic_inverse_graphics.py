from inverse_graphics_base import InverseGraphicsBase
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os, os.path


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
            newArr = np.where(arr > 0, 1, arr)
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

        if self.depth is None:
            print("converting rgbd to point cloud")
            input_image = self.convertImage2RGBD(self.images, self.depth)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                image=input_image,
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    width=np.asarray(input_image.color).shape[0],
                    height=np.asarray(input_image.color).shape[1],
                    intrinsic_matrix=self.intrinsics,
                ),
                extrinsic=self.extrinsics,
            )
        else:
            print("converting depth to point cloud")
            input_image = self.convertImage2RGBD(self.images, self.depth)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                image=input_image,
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    width=np.asarray(input_image.color).shape[0],
                    height=np.asarray(input_image.color).shape[1],
                    intrinsic_matrix=self.intrinsics,
                ),
                extrinsic=self.extrinsics,
            )

        pcd.transform(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )  # Flip it, otherwise the pointcloud will be upside down

        pcd.estimate_normals()

        return pcd

    def convertImage2PCWMask(self, mask) -> o3d.geometry.PointCloud:
        """
        Converts RGBD or Depth and mask images into PointClouds. Takes into account camera intrinsics and extrinsics

        Args:
            mask (numpy array): The array of the mask image.

        Returns:
            open3d PointCloud.
        """

        mask = self.checkMaskMax(mask)
        mask_3d = np.stack((mask, mask, mask), axis=2)
        input_rgb = self.images * mask_3d
        input_depth = self.depth * mask

        input_image = self.convertImage2RGBD(np.ascontiguousarray(input_rgb), np.ascontiguousarray(input_depth))

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=input_image,
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=np.asarray(input_image.color).shape[0],
                height=np.asarray(input_image.color).shape[1],
                intrinsic_matrix=self.intrinsics,
            ),
            extrinsic=self.extrinsics,
        )

        pcd.transform(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )  # Flip it, otherwise the pointcloud will be upside down

        pcd.estimate_normals()

        return pcd

    def run(self, pointCloud=None) -> Tuple[o3d.geometry.TriangleMesh, o3d.utility.DoubleVector]:
        """
        Converts PointClouds into mesh and density with Poisson surface recontruction

        Returns:
            Number of triangle mesh and the density.
        """
        print("run Poisson surface reconstruction")
        if pointCloud is None:
            print("converting image to point cloud")
            pcd = self.convertImage2PC()
        else:
            pcd = pointCloud

        print("creating triangle mesh")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        return mesh, densities


def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_coarse,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine, icp_fine.transformation
    )
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id],
                pcds[target_id],
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine,
            )
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=False,
                    )
                )
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True,
                    )
                )
    return pose_graph


def main():
    sample_path = "../../data/sample_data/"
    image_path = sample_path + "images/image0000.png"
    depth_path = sample_path + "depths/depth0000.txt"
    mask_path = sample_path + "binary_masks/mask0000.png"
    intrinsic_path = sample_path + "intrinsics/intrinsics0000.txt"
    extrinsic_path = sample_path + "camera_poses/pose0000.txt"
    colourRaw = np.asarray(o3d.io.read_image(image_path))
    depthRaw = np.loadtxt(depth_path)
    maskImg = np.asarray(o3d.io.read_image(mask_path))
    intrinsic = np.loadtxt(intrinsic_path)
    extrinsic = np.loadtxt(extrinsic_path)

    basicGraph = BasicInverseGraph(colourRaw, intrinsic, extrinsic, depthRaw)
    pcd = basicGraph.convertImage2PCWMask(maskImg)
    # mesh, densities = basicGraph.run(pcd)


if __name__ == "__main__":
    main()
