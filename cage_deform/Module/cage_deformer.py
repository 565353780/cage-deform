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

    def deformPoints(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
        deform_point_idxs: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        lr = 1e-2
        steps = 500
        lambda_reg = 1e4 # 平滑项权重，越大越刚性/平滑

        points = toTensor(points, self.dtype, self.device)
        deform_point_idxs = toTensor(deform_point_idxs, torch.int32, self.device)
        target_points = toTensor(target_points, self.dtype, self.device)

        print(f"Total points: {points.shape[0]}, Control points: {deform_point_idxs.shape[0]}")

        # 初始化变形器（现在需要传入三个参数）
        deformer = FFD(
            points,
            deform_point_idxs,
            target_points,
            voxel_size=1.0 / 8,
            padding=0.1,
        ).to(self.device)
        optimizer = optim.Adam(deformer.parameters(), lr=lr)

        for i in range(steps):
            optimizer.zero_grad()

            # forward现在直接返回loss
            loss_dict = deformer(lambda_reg=lambda_reg)

            loss = loss_dict['Loss']
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                loss_fit = loss_dict['LossFit']
                loss_reg = loss_dict['LossReg']
                print(f"Step {i}: Fit Loss = {loss_fit.item():.6f}, Reg Loss = {loss_reg.item():.6f}, Total Loss = {loss.item():.6f}")

        # 使用toWorldDeformedPoints获取最终结果
        final_points = deformer.toWorldDeformedPoints()

        print("Optimization Done.")
        return final_points
