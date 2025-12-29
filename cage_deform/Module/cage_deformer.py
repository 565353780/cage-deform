import torch
import numpy as np
import torch.optim as optim
from typing import Union

from cage_deform.Model.ffd import FFD
from cage_deform.Method.data import toTensor


class CageDeformer(object):
    def __init__(
        self,
        dtype=torch.float32,
        device: str = 'cpu',
    ) -> None:
        self.dtype = dtype
        self.device = device
        return

    def matchPoints(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
        deform_point_idxs: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        points = toTensor(points, self.dtype, self.device)
        deform_point_idxs = toTensor(deform_point_idxs, torch.int64, self.device)
        target_points = toTensor(target_points, self.dtype, self.device)

        # 取控制点及其目标位置（都应为 [N, 3]）
        src = points[deform_point_idxs]   # [N, 3]
        tgt = target_points               # [N, 3]
        assert src.shape == tgt.shape and src.ndim == 2 and src.shape[1] == 3

        # 求中心化
        src_mean = src.mean(dim=0, keepdim=True)
        tgt_mean = tgt.mean(dim=0, keepdim=True)

        src_centered = src - src_mean     # [N,3]
        tgt_centered = tgt - tgt_mean

        # 求仿射变换: 最小二乘拟合 Y = X @ A^T + t
        # src: [N,3], tgt: [N,3]
        # 解 A, t
        X = src_centered.cpu().numpy()
        Y = tgt_centered.cpu().numpy()
        # 最小二乘解: A^T = (X^T X)^-1 X^T Y
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        # A: [3,3]，表示src_center乘A ≈ tgt_center

        # 由于 A^T = ...，这里A是3x3（行表示src的三个分量，列对应tgt的分量）
        # 构造完整仿射矩阵
        affine = np.eye(4, dtype=np.float32)
        affine[:3,:3] = A.T
        t = (tgt_mean - src_mean @ torch.from_numpy(A).to(src_mean.device)).squeeze(0).cpu().numpy()
        affine[:3,3] = t

        # 对全体点应用该仿射变换
        points_np = points.cpu().numpy()
        points_h = np.concatenate([points_np, np.ones((points_np.shape[0], 1), dtype=np.float32)], axis=1) # [N,4]
        deformed_points_np = (points_h @ affine.T)[:, :3]
        matched_points = toTensor(deformed_points_np, self.dtype, self.device)

        return matched_points

    def deformPoints(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
        deform_point_idxs: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        voxel_size: float = 1.0 / 8,
        padding: float = 0.1,
        lr: float = 1e-2,
        lambda_reg: float = 1e4,
        steps: int = 500,
    ) -> torch.Tensor:
        # 确保梯度计算已启用（防止Detector初始化时可能禁用了梯度）
        torch.set_grad_enabled(True)

        points = toTensor(points, self.dtype, self.device)
        deform_point_idxs = toTensor(deform_point_idxs, torch.int64, self.device)
        target_points = toTensor(target_points, self.dtype, self.device)

        matched_points = self.matchPoints(points, deform_point_idxs, target_points)

        print(f"Total points: {points.shape[0]}, Control points: {deform_point_idxs.shape[0]}")

        deformer = FFD(
            matched_points,
            deform_point_idxs,
            target_points,
            voxel_size=voxel_size,
            padding=padding,
        ).to(self.device)
        # 确保模型处于训练模式，以便梯度计算
        deformer.train()
        # 确保所有参数都启用梯度
        for param in deformer.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(deformer.parameters(), lr=lr)

        for i in range(steps):
            optimizer.zero_grad()

            loss_dict = deformer(lambda_reg=lambda_reg)

            loss = loss_dict['Loss']
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                loss_fit = loss_dict['LossFit']
                loss_reg = loss_dict['LossReg']
                print(f"Step {i}: Fit Loss = {loss_fit.item():.6f}, Reg Loss = {loss_reg.item():.6f}, Total Loss = {loss.item():.6f}")

        final_points = deformer.toWorldDeformedPoints()

        print("Optimization Done.")
        return final_points
