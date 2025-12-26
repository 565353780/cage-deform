import torch
import open3d as o3d
from cage_deform.Method.data import toTensor

from cage_deform.Method.io import loadMeshFile
from cage_deform.Method.mesh import toO3DMesh
from cage_deform.Module.cage_deformer import CageDeformer


def demo():
    mesh_file_path = "/Users/chli/Downloads/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4afv1_5_-1_stagetwo_1024.glb"
    dtype = torch.float32
    device = 'cpu'

    mesh = loadMeshFile(mesh_file_path)

    if mesh is None:
        print('[ERROR][demo]')
        print('\t loadMeshFile failed!')
        return False

    points = toTensor(mesh.vertices, dtype, device)
    deform_point_idxs = torch.range(0, 10, 1, dtype=torch.int32, device=device)
    target_points = points[deform_point_idxs] + torch.tensor([0, 0, 10], dtype=dtype, device=device)

    cage_deformer = CageDeformer(dtype, device)

    deformed_points = cage_deformer.deformPoints(points, deform_point_idxs, target_points)

    source_mesh = toO3DMesh(mesh.vertices, mesh.faces)
    target_mesh = toO3DMesh(deformed_points, mesh.faces)

    o3d.visualization.draw_geometries([source_mesh, target_mesh])
