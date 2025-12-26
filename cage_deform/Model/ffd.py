import torch
import torch.nn as nn
import torch.nn.functional as F

class FFD(nn.Module):
    def __init__(self, points, voxel_size=0.1, padding=0.1):
        """
        初始化可微FFD变形器
        :param points: 初始点云 (N, 3) tensor
        :param voxel_size: 笼子每个格子的边长
        :param padding: 笼子包裹点云时的额外边界比例
        """
        super().__init__()
        self.device = points.device
        self.initial_points = points.clone()

        # 1. 计算点云的边界框
        self.min_coords = points.min(dim=0)[0] - padding
        self.max_coords = points.max(dim=0)[0] + padding
        self.extent = self.max_coords - self.min_coords

        # 2. 根据voxel_size计算笼子的维度 (Depth, Height, Width)
        # 注意：Grid Sample需要的顺序通常是 (D, H, W) 对应 (z, y, x)
        dims = (self.extent / voxel_size).ceil().int()
        self.grid_D, self.grid_H, self.grid_W = dims[2].item(), dims[1].item(), dims[0].item()

        print(f"Cage Grid Size: {self.grid_W}x{self.grid_H}x{self.grid_D} (XxYxZ)")

        # 3. 初始化笼子顶点的偏移量 (Learnable Parameter)
        # 形状为 (1, 3, D, H, W)，表示每个格点的 (dx, dy, dz)
        # 初始化为0，表示初始状态无形变
        self.cage_offsets = nn.Parameter(torch.zeros(1, 3, self.grid_D, self.grid_H, self.grid_W).to(self.device))

        # 4. 预计算点云在Grid Sample中的归一化坐标 [-1, 1]
        self.norm_coords = self._normalize_coords(points)
        return

    def _normalize_coords(self, points):
        """将世界坐标转换为grid_sample需要的 [-1, 1] 坐标"""
        # 归一化到 [0, 1]
        norm = (points - self.min_coords) / self.extent
        # 归一化到 [-1, 1]
        norm = norm * 2 - 1

        # Grid Sample 需要形状 (1, 1, 1, N, 3) 
        # 且最后一维顺序应为 (x, y, z)
        return norm.view(1, 1, 1, -1, 3)

    def forward(self):
        """
        前向传播：计算变形后的点云
        """
        # 使用 grid_sample 进行三线性插值
        # input: (N, C, D_in, H_in, W_in) -> 我们的 cage_offsets
        # grid: (N, D_out, H_out, W_out, 3) -> 我们的点云坐标
        # align_corners=True 保证 -1 和 1 精确映射到笼子的边界

        # 采样出每个点的位移量
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

        # 应用位移
        deformed_points = self.initial_points + point_offsets
        return deformed_points

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
