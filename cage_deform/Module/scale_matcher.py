import os
import torch
import numpy as np
import open3d as o3d

from tqdm import trange
from typing import Tuple

if torch.cuda.is_available():
    from cage_deform.Lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist
else:
    from cage_deform.Lib.chamfer3D.chamfer_python import distChamfer

from cage_deform.Method.io import loadMeshFile
from cage_deform.Method.path import createFileFolder
from cage_deform.Method.data import toNumpy, toTensor


def vec6_to_symmat(v):
    """ [6] -> symmetric 3x3，用 stack 保持梯度 """
    row0 = torch.stack([v[0], v[1], v[2]])
    row1 = torch.stack([v[1], v[3], v[4]])
    row2 = torch.stack([v[2], v[4], v[5]])
    return torch.stack([row0, row1, row2])

def so3_exp(w):
    """ axis-angle -> rotation matrix，用 stack 保持梯度 """
    theta = torch.norm(w) + 1e-8
    k = w / theta
    z = torch.zeros(1, device=w.device, dtype=w.dtype).squeeze(0)
    K = torch.stack([
        torch.stack([z, -k[2], k[1]]),
        torch.stack([k[2], z, -k[0]]),
        torch.stack([-k[1], k[0], z]),
    ])
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

class ScaleMatcher(object):
    def __init__(
        self,
        device: str='cuda:0',
    ) -> None:
        self.device = device

        if torch.cuda.is_available() and device != "cpu":
            self.chamfer_func = chamfer_3DDist()
        else:
            self.chamfer_func = distChamfer

        self.h_param = torch.zeros(6, device=self.device, requires_grad=True)  # 对称矩阵
        self.r_param = torch.zeros(3, device=self.device, requires_grad=True)  # axis-angle
        self.t_param = torch.zeros(3, device=self.device, requires_grad=True)  # translation
        return

    def reset(self) -> bool:
        self.h_param = torch.zeros(6, device=self.device, requires_grad=True)  # 对称矩阵
        self.r_param = torch.zeros(3, device=self.device, requires_grad=True)  # axis-angle
        self.t_param = torch.zeros(3, device=self.device, requires_grad=True)  # translation
        return True

    def toTransform(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---- SPD scale ----
        H = vec6_to_symmat(self.h_param)
        S = torch.matrix_exp(H)  # SPD

        # ---- Rotation ----
        R = so3_exp(self.r_param)

        # ---- 4x4 变换矩阵（行向量右乘：p' = p @ T，无需转置） ----
        # A 为行空间 3x3：p' = p @ A + t => A = S.T @ R.T
        A = S.T @ R.T
        T = torch.eye(4, device=self.device)
        T[:3, :3] = A
        T[:3, 3] = self.t_param

        return T, H

    def matchScale(
        self,
        source_points,
        target_points,
        lr=1e-3,
        steps=1000,
    ) -> np.ndarray:
        self.reset()

        optimizer = torch.optim.AdamW(
            [
                self.h_param,
                self.r_param,
                self.t_param,
            ],
            lr=lr,
        )

        source_pts = toTensor(source_points, device=self.device).reshape(-1, 3)
        target_pts = toTensor(target_points, device=self.device).reshape(1, -1, 3)

        ones = torch.ones(source_pts.shape[0], 1, device=self.device)
        X_h = torch.cat([source_pts, ones], dim=1)  # [N,4]
        X = X_h.unsqueeze(0) # [1,N,4]

        pbar = trange(steps, desc="matchScale")
        for i in pbar:
            T, H = self.toTransform()

            # ---- 右乘：X_h [N,4] @ T [4,4] ----
            Xp = (X @ T)[..., :3]

            # ---- Chamfer ----
            dist1, dist2 = self.chamfer_func(Xp, target_pts)[:2]
            chamfer_loss = dist1.mean() + dist2.mean()

            # ---- Regularization (VERY important) ----
            scale_reg = (H ** 2).mean()          # 防止 scale 爆炸
            trans_reg = (self.t_param ** 2).mean()

            loss = chamfer_loss + 0.01 * scale_reg + 0.001 * trans_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.6f}")

        # 返回 4x4 变换矩阵，可与 Nx4 点云右乘：points_transformed = points @ T
        with torch.no_grad():
            T = self.toTransform()[0]

        return toNumpy(T, np.float32)

    def matchScaleFile(
        self,
        source_mesh_file_path: str,
        target_pcd_file_path: str,
        save_mesh_file_path: str,
        lr=1e-3,
        steps=1000,
    ) -> bool:
        if os.path.exists(save_mesh_file_path):
            return True

        source_mesh = loadMeshFile(source_mesh_file_path)
        if source_mesh is None:
            print('[ERROR][ScaleMatcher::matchScaleFile]')
            print('\t loadMeshFile failed!')
            return False

        target_pcd = o3d.io.read_point_cloud(target_pcd_file_path)

        source_points = np.asarray(source_mesh.vertices)
        target_points = np.asarray(target_pcd.points)

        T = self.matchScale(
            source_points,
            target_points,
            lr,
            steps,
        )
        # 点云 (N,4) 右乘 T
        ones = np.ones((len(source_points), 1), dtype=source_points.dtype)
        source_h = np.hstack([source_points, ones])  # (N, 4)
        aligned = (source_h @ T)[:, :3]
        source_mesh.vertices = aligned

        createFileFolder(save_mesh_file_path)
        source_mesh.export(save_mesh_file_path)
        return True
