from typing import List

import numpy as np
import open3d as o3d

try:
    from learning_real2sim.src.meta_ball import MetaBall
except:
    print(
        "Failed to import learning_real2sim. FuzzyMetaballMeshProcessor won't be working!"
    )

from sim2sim.logging import DynamicLogger
from sim2sim.util import MeshProcessorResult, open3d_to_trimesh

from .mesh_processor_base import MeshProcessorBase

DIV_EPSILON = 1e-9


class FuzzyMetaballMeshProcessor(MeshProcessorBase):
    """
    Replaces the mesh with ellipsoids obtained from fuzzy metaballs.
    https://leonidk.github.io/fuzzy-metaballs/
    """

    def __init__(
        self,
        logger: DynamicLogger,
        visualize: bool,
        mesh_paths: List[str],
        num_iter: int,
        gmm_init: bool,
        remove_outliers: bool,
        normalize_mesh: bool,
    ):
        """
        :param logger: The logger.
        :param visualize: Whether to visualize the fitted spheres.
        :param mesh_paths: The paths to the meshes to process. NOTE: This is used
            instead of the `mesh` argument of `process_mesh`.
        :param num_iter: The number of fuzzy metaball gradient descent iterations.
        :param gmm_init: Whether to initialize the metaballs using GMM EM or random.
        :param remove_outliers: Whether to remove metaball ellipsoid outliers.
        :param normalize_mesh: Whether to normalize the mesh before optimization.
        """
        super().__init__(logger)

        self._visualize = visualize
        self._mesh_paths = mesh_paths
        self._num_iter = num_iter
        self._gmm_init = gmm_init
        self._remove_outliers = remove_outliers
        self._normalize_mesh = normalize_mesh

    def process_meshes(
        self, meshes: List[o3d.geometry.TriangleMesh]
    ) -> List[MeshProcessorResult]:
        mesh_processor_results = []
        for mesh_path in self._mesh_paths:
            meta_ball = MetaBall.generate_metaballs_from_mesh(
                mesh_path,
                iter_num=self._num_iter,
                gmm_init=self._gmm_init,
                normalized=self._normalize_mesh,
            )

            if self._visualize:
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                meta_ball.vis(idx=0, mesh=open3d_to_trimesh(mesh), remove_outliers=True)

            ellipsoids = meta_ball.to_analytical_ellipsoids(
                shape_idx=0, remove_outliers=True
            )
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

            mesh_processor_results.append(
                MeshProcessorResult(
                    result_type=MeshProcessorResult.ResultType.PRIMITIVE_INFO,
                    primitive_info=primitive_info,
                )
            )

        return mesh_processor_results
