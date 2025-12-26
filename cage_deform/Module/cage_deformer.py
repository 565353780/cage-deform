import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
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

    def deformPoints(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
        deform_point_idxs: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        lr = 1e-2
        steps = 500
        lambda_reg = 10.0 # 平滑项权重，越大越刚性/平滑

        points = toTensor(points, self.dtype, self.device)
        deform_point_idxs = toTensor(deform_point_idxs, torch.int32, self.device)
        target_points = toTensor(target_points, self.dtype, self.device)

        print(f"Total points: {points.shape[0]}, Control points: {deform_point_idxs.shape[0]}")

        # 3. 初始化变形器
        deformer = FFD(points, voxel_size=0.3, padding=0.2).to(self.device)
        optimizer = optim.Adam(deformer.parameters(), lr=lr)

        for i in range(steps):
            optimizer.zero_grad()

            current_points = deformer()

            fitting_points = current_points[deform_point_idxs]
            loss_fit = F.mse_loss(fitting_points, target_points)

            loss_reg = deformer.get_regularization_loss()

            loss = loss_fit + lambda_reg * loss_reg

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Step {i}: Fit Loss = {loss_fit.item():.6f}, Reg Loss = {loss_reg.item():.6f}")

        final_points = deformer()

        print("Optimization Done.")
        return final_points
