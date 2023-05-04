from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Any, Dict

import open3d as o3d

from sim2sim.logging import DynamicLogger


class MeshProcessorBase(ABC):
    """
    The mesh processor responsible for postprocessing meshes produced from inverse graphics before adding them to the
    simulation.
    """

    def __init__(self, logger: DynamicLogger):
        """
        :param logger: The logger.
        """
        self._logger = logger

    @abstractmethod
    def process_mesh(
        self, mesh: o3d.geometry.TriangleMesh
    ) -> Tuple[
        bool,
        Union[o3d.geometry.TriangleMesh, None],
        List[o3d.geometry.TriangleMesh],
        Union[List[Dict[str, Any]], None],
        Union[str, None],
    ]:
        """
        :param mesh: The mesh to process.
        :return: A tuple of (is_primitive, mesh, mesh_parts, primitive_info):
            - is_primitive: Whether it is a mesh or primitives.
            - mesh, mesh_parts: Either a single mesh or a list of mesh parts that form the mesh. TODO: Replace this with
                a single list parameter.
            - primitive_info: A list of dicts containing primitive params. Each dict must contain "name" which can for
                example be sphere, ellipsoid, box, etc. and "transform" which is a homogenous transformation matrix. The
                other params are primitive dependent but must be sufficient to construct that primitive. TODO: Create an
                enum type for "name".
            - sdf_path: A path to an SDFormat file to use directly. The presence of this overrides everything else.
        """
        raise NotImplementedError
