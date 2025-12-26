import torch
import numpy as np
import open3d as o3d
from typing import Union

from cage_deform.Method.data import toNumpy

def toO3DPcd(
    points: Union[torch.Tensor, np.ndarray, list],
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(toNumpy(points, np.float64))
    return pcd

def toO3DMesh(
    vertices: Union[torch.Tensor, np.ndarray, list],
    triangles: Union[torch.Tensor, np.ndarray, list],
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(toNumpy(vertices, np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(toNumpy(triangles, np.int32))

    mesh.compute_vertex_normals()
    return mesh
