from typing import List, Tuple

import numpy as np

from pydrake.all import DiagramBuilder, SceneGraph

from sim2sim.logging import DynamicLogger

from .image_generator_base import ImageGeneratorBase


class NoneImageGenerator(ImageGeneratorBase):
    """
    An image generator that returns empty lists instead of camera data.
    """

    def __init__(
        self,
        builder: DiagramBuilder,
        scene_graph: SceneGraph,
        logger: DynamicLogger,
    ):
        super().__init__(builder, scene_graph, logger)

    def generate_images(
        self,
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        camera_poses = []
        intrinsics = []
        images = []
        depths = []
        labels = []
        masks = []
        self._logger.log(
            camera_poses=camera_poses,
            intrinsics=intrinsics,
            images=images,
            depths=depths,
            labels=labels,
            masks=masks,
        )

        return (
            images,
            intrinsics,
            camera_poses,
            depths,
            labels,
            masks,
        )
