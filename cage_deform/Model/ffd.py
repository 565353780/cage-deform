import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union


class FFD(nn.Module):
    def __init__(
        self, 
        points: Union[torch.Tensor, np.ndarray, list],
        deform_point_idxs: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        voxel_size=0.1, 
        padding=0.1
    ):
        """
        初始化可微FFD变形器
        :param points: 初始点云 (N, 3) tensor (世界坐标系)
        :param deform_point_idxs: 需要变形的点的索引 (M,) tensor
        :param target_points: 目标点云 (M, 3) tensor (世界坐标系)
        :param voxel_size: 笼子每个格子的边长
        :param padding: 笼子包裹点云时的额外边界比例
        """
        super().__init__()
        self.device = points.device

        # 1. 计算点云的边界框和归一化参数（基于points）
        min_coords = points.min(dim=0)[0]
        max_coords = points.max(dim=0)[0]
        extent = max_coords - min_coords

        # 计算center和scale（scale是最大边长的一半，用于归一化到[-0.5, 0.5]）
        center = (min_coords + max_coords) / 2.0  # 中心点
        scale = extent.max() / 2.0  # 最大边长的一半

        # 注册为buffer，这样在模型保存/加载时会被正确处理
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)

        # 2. 使用相同的变换归一化points和target_points到[-0.5, 0.5]
        normalized_points = (points - self.center) / self.scale
        normalized_target_points = (target_points - self.center) / self.scale

        # 保存归一化后的初始点云和目标点云
        self.initial_points = normalized_points.clone()
        self.register_buffer('normalized_target_points', normalized_target_points)
        self.register_buffer('deform_point_idxs', deform_point_idxs.long())

        # 3. 计算带padding的边界框（用于构建笼子）
        # padding是相对于归一化后点云范围的比例
        # 归一化后的点云范围大约是[-0.5, 0.5]，即范围是1
        normalized_min = normalized_points.min(dim=0)[0]
        normalized_max = normalized_points.max(dim=0)[0]
        normalized_extent = normalized_max - normalized_min
        padding_size = normalized_extent * padding

        self.min_coords = normalized_min - padding_size
        self.max_coords = normalized_max + padding_size
        self.extent = self.max_coords - self.min_coords

        # 4. 根据voxel_size计算笼子的维度 (Depth, Height, Width)
        # 注意：Grid Sample需要的顺序通常是 (D, H, W) 对应 (z, y, x)
        # voxel_size也需要归一化到归一化空间
        normalized_voxel_size = voxel_size / self.scale
        dims = (self.extent / normalized_voxel_size).ceil().int()
        self.grid_D, self.grid_H, self.grid_W = dims[2].item(), dims[1].item(), dims[0].item()

        print(f"Cage Grid Size: {self.grid_W}x{self.grid_H}x{self.grid_D} (XxYxZ)")
        print(f"Normalization: center={self.center.cpu().numpy()}, scale={self.scale.item():.6f}")

        # 5. 初始化笼子顶点的偏移量 (Learnable Parameter)
        # 形状为 (1, 3, D, H, W)，表示每个格点的 (dx, dy, dz)
        # 初始化为0，表示初始状态无形变
        self.cage_offsets = nn.Parameter(torch.zeros(1, 3, self.grid_D, self.grid_H, self.grid_W).to(self.device))

        # 6. 预计算点云在Grid Sample中的归一化坐标 [-0.5, 0.5]
        self.norm_coords = self._normalize_coords(normalized_points)
        return

    def _normalize_coords(self, points):
        """将归一化坐标（[-0.5, 0.5]）转换为grid_sample需要的 [-1, 1] 坐标"""
        # 归一化到 [0, 1]
        norm = (points - self.min_coords) / self.extent
        # 归一化到 [-1, 1]
        norm = norm * 2 - 1

        # Grid Sample 需要形状 (1, 1, 1, N, 3) 
        # 且最后一维顺序应为 (x, y, z)
        return norm.view(1, 1, 1, -1, 3)

    def forward(self, lambda_reg: float=10.0) -> dict:
        """
        前向传播：计算loss
        :param lambda_reg: 正则化项权重
        :return: 总loss
        """
        # 使用 grid_sample 进行三线性插值
        # input: (N, C, D_in, H_in, W_in) -> 我们的 cage_offsets
        # grid: (N, D_out, H_out, W_out, 3) -> 我们的点云坐标
        # align_corners=True 保证 -1 和 1 精确映射到笼子的边界

        # 采样出每个点的位移量（在归一化空间中）
        point_offsets = F.grid_sample(
            self.cage_offsets, 
            self.norm_coords, 
            mode='bilinear', # 3D下即三线性插值
            padding_mode='border',
            align_corners=True
        )

        # 输出形状调整回 (N, 3)
        # grid_sample输出为 (1, 3, 1, 1, N)，需要permute
        point_offsets = point_offsets.view(3, -1).permute(1, 0)

        # 应用位移（在归一化空间中）
        deformed_points_normalized = self.initial_points + point_offsets

        # 计算fitting loss（在归一化空间中）
        fitting_points = deformed_points_normalized[self.deform_point_idxs]
        loss_fit = F.smooth_l1_loss(fitting_points, self.normalized_target_points)

        # 计算正则化loss
        loss_reg = self.get_regularization_loss()

        # 总loss
        loss = loss_fit + lambda_reg * loss_reg

        loss_dict = {
            'Loss': loss,
            'LossFit': loss_fit,
            'LossReg': loss_reg,
        }

        return loss_dict

    def toWorldDeformedPoints(self):
        """
        获取变形后的点云（世界坐标系）
        :return: 变形后的点云 (N, 3) tensor (世界坐标系)
        """
        # 使用 grid_sample 进行三线性插值
        point_offsets = F.grid_sample(
            self.cage_offsets, 
            self.norm_coords, 
            mode='bilinear',
            padding_mode='border', 
            align_corners=True
        )

        # 输出形状调整回 (N, 3)
        point_offsets = point_offsets.view(3, -1).permute(1, 0)

        # 应用位移（在归一化空间中）
        deformed_points_normalized = self.initial_points + point_offsets

        # 反归一化回世界坐标系
        deformed_points_world = deformed_points_normalized * self.scale + self.center

        return deformed_points_world

    def get_regularization_loss(self):
        """
        计算平滑约束 (Laplacian Smoothness)
        确保笼子的变形是平滑的，模拟刚性/弹性物体
        """
        grid = self.cage_offsets

        # 计算二阶差分 (Laplacian近似)
        # 沿D轴 (Z)
        dzz = grid[:, :, 2:, :, :] - 2 * grid[:, :, 1:-1, :, :] + grid[:, :, :-2, :, :]
        # 沿H轴 (Y)
        dyy = grid[:, :, :, 2:, :] - 2 * grid[:, :, :, 1:-1, :] + grid[:, :, :, :-2, :]
        # 沿W轴 (X)
        dxx = grid[:, :, :, :, 2:] - 2 * grid[:, :, :, :, 1:-1] + grid[:, :, :, :, :-2]

        loss_smooth = torch.mean(dzz**2) + torch.mean(dyy**2) + torch.mean(dxx**2)

        # 可选：加一个极小的L2正则，防止漂移过远
        loss_magnitude = torch.mean(grid**2)

        return loss_smooth + 0.01 * loss_magnitude
