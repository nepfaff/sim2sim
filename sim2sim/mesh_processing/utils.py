import open3d as o3d
import numpy as np
import trimesh


def open3d_to_trimesh(src):

    """Convert mesh from open3d to trimesh
    https://github.com/wkentaro/morefusion/blob/b8b892b3fbc384982a4929b1418ee29393069b11/morefusion/utils/open3d_to_trimesh.py
    """
    if isinstance(src, o3d.geometry.TriangleMesh):
        vertex_colors = None
        if src.has_vertex_colors:
            vertex_colors = np.asarray(src.vertex_colors)
        dst = trimesh.Trimesh(
            vertices=np.asarray(src.vertices),
            faces=np.asarray(src.triangles),
            vertex_normals=np.asarray(src.vertex_normals),
            vertex_colors=vertex_colors,
        )
    else:
        raise ValueError("Unsupported type of src: {}".format(type(src)))

    return dst
