import torch
import trimesh
import numpy as np

from cage_deform.Method.data import toTensor

from cage_deform.Module.bspline_deformer import BSplineDeformer


def demo():
    mesh: trimesh.Trimesh
    source_points: torch.Tensor
    target_points: torch.Tensor
    voxel_size = 1.0 / 64
    padding = 0.1
    lr = 1e-2
    lambda_smooth: float = 1e3
    lambda_magnitude: float = 1.0
    steps = 1000
    dtype = torch.float32
    device: str = 'cpu'
    vertices = toTensor(mesh.vertices, dtype, device)

    bspline_deformer = BSplineDeformer(dtype, device)

    bspline_deformer.loadPoints(mesh.vertices, voxel_size, padding)

    deformed_points = bspline_deformer.deformPoints(
        source_points, target_points,
        lr, lambda_smooth, lambda_magnitude, steps,
    )

    deformed_vertices = bspline_deformer.queryPoints(vertices)

    deformed_trimesh = deepcopy(mesh)
    deformed_trimesh.vertices = toNumpy(deformed_vertices, np.float32)
    return True
