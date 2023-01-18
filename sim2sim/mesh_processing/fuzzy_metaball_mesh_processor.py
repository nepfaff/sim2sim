from typing import Tuple, List, Union, Any, Dict

import open3d as o3d
import numpy as np

from .mesh_processor_base import MeshProcessorBase
from sim2sim.logging import DynamicLoggerBase
from learning_real2sim.src.meta_ball import MetaBall
from sim2sim.util import open3d_to_trimesh

DIV_EPSILON = 1e-9


class FuzzyMetaballMeshProcessor(MeshProcessorBase):
    """Replaces the mesh with ellipsoids obtained from fuzzy metaballs. https://leonidk.github.io/fuzzy-metaballs/"""

    def __init__(
        self,
        logger: DynamicLoggerBase,
        visualize: bool,
        mesh_path: str,
        num_iter: int,
        gmm_init: bool,
        remove_outliers: bool,
        normalize_mesh: bool,
    ):
        """
        :param logger: The logger.
        :param visualize: Whether to visualize the fitted spheres.
        :param mesh_path: The path of the mesh to process. NOTE: This is used instead of the `mesh` argument of
            `process_mesh`.
        :param num_iter: The number of fuzzy metaball gradient descent iterations.
        :param gmm_init: Whether to initialize the metaballs using GMM EM or random.
        :param remove_outliers: Whether to remove metaball ellipsoid outliers.
        :param normalize_mesh: Whether to normalize the mesh before optimization.
        """
        super().__init__(logger)

        self._visualize = visualize
        self._mesh_path = mesh_path
        self._num_iter = num_iter
        self._gmm_init = gmm_init
        self._remove_outliers = remove_outliers
        self._normalize_mesh = normalize_mesh

    def process_mesh(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> Tuple[
        bool,
        Union[o3d.geometry.TriangleMesh, None],
        List[o3d.geometry.TriangleMesh],
        Union[List[Dict[str, Any]], None],
    ]:
        meta_ball = MetaBall.generate_metaballs_from_mesh(
            self._mesh_path, iter_num=self._num_iter, gmm_init=self._gmm_init, normalized=self._normalize_mesh
        )

        if self._visualize:
            meta_ball.vis(idx=0, mesh=open3d_to_trimesh(mesh), remove_outliers=True)

        ellipsoids = meta_ball.to_analytical_ellipsoids(shape_idx=0, remove_outliers=True)
        primitive_info = []
        for ellipsoid in ellipsoids:
            _, radii, transform = ellipsoid
            primitive_info.append(
                {
                    "name": "ellipsoid",
                    "radii": np.asarray(radii),
                    "transform": np.asarray(transform),
                }
            )

        return True, None, [], primitive_info
