import os

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from pydrake.all import (
    DiagramBuilder,
    MakeRenderEngineGl,
    RenderEngineGlParams,
    SceneGraph,
)

from sim2sim.logging import DynamicLogger


class ImageGeneratorBase(ABC):
    """The image generator responsible for placing cameras and generating camera data."""

    def __init__(
        self, builder: DiagramBuilder, scene_graph: SceneGraph, logger: DynamicLogger
    ):
        """
        :param builder: The diagram builder.
        :param scene_graph: The scene graph.
        :param logger: The logger.
        """
        self._builder = builder
        self._scene_graph = scene_graph
        self._logger = logger
        self._diagram = None

        # Add renderer
        have_display = bool(os.environ.get("DISPLAY", None))
        if not have_display:
            print("No display available!")
        self._renderer = "ImageGeneratorRenderer"
        if have_display and not self._scene_graph.HasRenderer(self._renderer):
            self._scene_graph.AddRenderer(
                self._renderer, MakeRenderEngineGl(RenderEngineGlParams())
            )

    # TODO: Refactor to return a dataclass instead of a tuple
    @abstractmethod
    def generate_images(
        self,
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        Union[List[np.ndarray], None],
        Union[List[np.ndarray], None],
    ]:
        """
        Generate images using existing or new cameras.

        :return: A tuple of (images, intrinsics, extrinsics, depths, masks):
            - images: The RGB images of shape (m,n,3) where m is the image height and n is the image width.
            - intrinsics: The intrinsic matrices associated with the images of shape (3,3).
            - extrinsics: The extrinsic matrices associated with the images of shape (4,4).
            - depths: The depth images associated with the images of shape (m,n) where m is the image height and n is
                the image width.
            - labels: The object detection labels. Each image may have multiple associated labels.
            - masks: The binary segmentation masks associated with the labels of type np.uint8 and shape (m,n) where m
                is the image height and n is the image width.
        """
        raise NotImplementedError
