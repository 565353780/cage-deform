import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from typing import Union, Optional

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

        # 体素网格相关属性
        self.voxel_size: float = 0.125
        self.padding: float = 0.1
        self.grid_W: int = 0
        self.grid_H: int = 0
        self.grid_D: int = 0

        # 归一化参数
        self.center: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.min_coords: Optional[torch.Tensor] = None
        self.max_coords: Optional[torch.Tensor] = None
        self.extent: Optional[torch.Tensor] = None

        # 体素顶点偏移量（形变参数）
        self.cage_offsets: Optional[torch.Tensor] = None

        # 初始点云（归一化空间）
        self.initial_points: Optional[torch.Tensor] = None
        return

    def loadPoints(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
        voxel_size: float = 0.125,
        padding: float = 0.1,
    ) -> dict:
        """
        加载点云并创建一个边长为 voxel_size 的 MxNxK 的体素集合覆盖住给定的点云。

        Args:
            points: 输入点云 (N, 3)
            voxel_size: 体素边长（世界坐标系）
            padding: 边界框的额外边界比例

        Returns:
            dict: 包含体素网格信息的字典
        """
        self.voxel_size = voxel_size
        self.padding = padding

        points = toTensor(points, self.dtype, self.device)

        # 1. 计算点云的边界框和归一化参数
        min_coords = points.min(dim=0)[0]
        max_coords = points.max(dim=0)[0]
        extent = max_coords - min_coords

        # 计算center和scale（scale是最大边长的一半，用于归一化到[-0.5, 0.5]）
        self.center = (min_coords + max_coords) / 2.0
        self.scale = extent.max() / 2.0

        # 2. 归一化点云到[-0.5, 0.5]
        normalized_points = (points - self.center) / self.scale
        self.initial_points = normalized_points.clone()

        # 3. 计算带padding的边界框（用于构建体素网格）
        normalized_min = normalized_points.min(dim=0)[0]
        normalized_max = normalized_points.max(dim=0)[0]
        normalized_extent = normalized_max - normalized_min
        padding_size = normalized_extent * padding

        self.min_coords = normalized_min - padding_size
        self.max_coords = normalized_max + padding_size
        self.extent = self.max_coords - self.min_coords

        # 4. 根据voxel_size计算体素网格的维度 (W, H, D) 对应 (x, y, z)
        normalized_voxel_size = voxel_size / self.scale
        dims = (self.extent / normalized_voxel_size).ceil().int()
        self.grid_W, self.grid_H, self.grid_D = dims[0].item(), dims[1].item(), dims[2].item()

        # 确保每个维度至少有2个格点
        self.grid_W = max(self.grid_W, 2)
        self.grid_H = max(self.grid_H, 2)
        self.grid_D = max(self.grid_D, 2)

        print(f"Voxel Grid Size: {self.grid_W}x{self.grid_H}x{self.grid_D} (WxHxD / XxYxZ)")
        print(f"Normalization: center={self.center.cpu().numpy()}, scale={self.scale.item():.6f}")

        # 5. 初始化体素顶点的偏移量（形变参数）
        # 形状为 (1, 3, D, H, W)，表示每个格点的 (dx, dy, dz)
        # 初始化为0，表示初始状态无形变
        self.cage_offsets = torch.zeros(
            1, 3, self.grid_D, self.grid_H, self.grid_W,
            dtype=self.dtype, device=self.device
        )

        # 返回体素网格信息
        return {
            'grid_size': (self.grid_W, self.grid_H, self.grid_D),
            'num_vertices': (self.grid_W + 1) * (self.grid_H + 1) * (self.grid_D + 1),
            'center': self.center.cpu().numpy(),
            'scale': self.scale.item(),
            'extent': self.extent.cpu().numpy(),
        }

    def _normalize_coords(self, points: torch.Tensor) -> torch.Tensor:
        """
        将世界坐标转换为 grid_sample 需要的 [-1, 1] 坐标。

        Args:
            points: 世界坐标系点云 (N, 3)

        Returns:
            归一化坐标 (1, 1, 1, N, 3)
        """
        # 先转换到归一化空间 [-0.5, 0.5]
        normalized_points = (points - self.center) / self.scale

        # 归一化到 [0, 1]
        norm = (normalized_points - self.min_coords) / self.extent
        # 归一化到 [-1, 1]
        norm = norm * 2 - 1

        # Grid Sample 需要形状 (1, 1, 1, N, 3)
        return norm.view(1, 1, 1, -1, 3)

    def _compute_affine_transform(
        self,
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
    ) -> np.ndarray:
        """
        计算从源点到目标点的仿射变换矩阵。

        Args:
            src_points: 源点云 (N, 3)
            tgt_points: 目标点云 (N, 3)

        Returns:
            4x4 仿射变换矩阵
        """
        src_mean = src_points.mean(dim=0, keepdim=True)
        tgt_mean = tgt_points.mean(dim=0, keepdim=True)

        src_centered = src_points - src_mean
        tgt_centered = tgt_points - tgt_mean

        X = src_centered.cpu().numpy()
        Y = tgt_centered.cpu().numpy()
        A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

        affine = np.eye(4, dtype=np.float32)
        affine[:3, :3] = A.T
        t = (tgt_mean - src_mean @ torch.from_numpy(A).to(src_mean.device)).squeeze(0).cpu().numpy()
        affine[:3, 3] = t

        return affine

    def matchPoints(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
        deform_point_idxs: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        """
        使用仿射变换将点云匹配到目标位置。

        Args:
            points: 输入点云 (N, 3)
            deform_point_idxs: 控制点索引 (M,)
            target_points: 目标点云 (M, 3)

        Returns:
            匹配后的点云 (N, 3)
        """
        points = toTensor(points, self.dtype, self.device)
        deform_point_idxs = toTensor(deform_point_idxs, torch.int64, self.device)
        target_points = toTensor(target_points, self.dtype, self.device)

        src = points[deform_point_idxs]
        tgt = target_points
        assert src.shape == tgt.shape and src.ndim == 2 and src.shape[1] == 3

        affine = self._compute_affine_transform(src, tgt)

        points_np = points.cpu().numpy()
        points_h = np.concatenate([points_np, np.ones((points_np.shape[0], 1), dtype=np.float32)], axis=1)
        deformed_points_np = (points_h @ affine.T)[:, :3]
        matched_points = toTensor(deformed_points_np, self.dtype, self.device)

        return matched_points

    def deformPoints(
        self,
        source_points: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        lr: float = 1e-2,
        lambda_reg: float = 1e4,
        steps: int = 500,
    ) -> torch.Tensor:
        """
        输入 source_points 和 target_points，估计体素顶点集合的形变，实现空间的自适应扭曲。
        
        需要先调用 loadPoints 加载点云并创建体素网格。

        Args:
            source_points: 源点云（世界坐标系）(M, 3)，这些点需要被形变到 target_points
            target_points: 目标点云（世界坐标系）(M, 3)
            lr: 学习率
            lambda_reg: 正则化权重
            steps: 优化步数

        Returns:
            形变后的源点云 (M, 3)
        """
        if self.cage_offsets is None:
            raise RuntimeError("请先调用 loadPoints 加载点云并创建体素网格！")

        # 确保梯度计算已启用
        torch.set_grad_enabled(True)

        source_points = toTensor(source_points, self.dtype, self.device)
        target_points = toTensor(target_points, self.dtype, self.device)

        print(f"Source points: {source_points.shape[0]}, Target points: {target_points.shape[0]}")

        # 1. 使用仿射变换进行初步对齐
        src_mean = source_points.mean(dim=0, keepdim=True)
        tgt_mean = target_points.mean(dim=0, keepdim=True)
        affine = self._compute_affine_transform(source_points, target_points)

        # 应用仿射变换到源点
        source_np = source_points.cpu().numpy()
        source_h = np.concatenate([source_np, np.ones((source_np.shape[0], 1), dtype=np.float32)], axis=1)
        aligned_source_np = (source_h @ affine.T)[:, :3]
        aligned_source = toTensor(aligned_source_np, self.dtype, self.device)

        # 2. 将源点和目标点归一化到体素网格空间
        normalized_source = (aligned_source - self.center) / self.scale
        normalized_target = (target_points - self.center) / self.scale

        # 3. 计算源点在体素网格中的归一化坐标
        norm_coords_source = self._normalize_coords(aligned_source)

        # 4. 初始化形变参数并设置为可学习
        self.cage_offsets = torch.zeros(
            1, 3, self.grid_D, self.grid_H, self.grid_W,
            dtype=self.dtype, device=self.device, requires_grad=True
        )

        optimizer = optim.Adam([self.cage_offsets], lr=lr)

        for i in range(steps):
            optimizer.zero_grad()

            # 使用 grid_sample 进行三线性插值获取每个点的偏移量
            point_offsets = F.grid_sample(
                self.cage_offsets,
                norm_coords_source,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            point_offsets = point_offsets.view(3, -1).permute(1, 0)

            # 应用偏移量
            deformed_points = normalized_source + point_offsets

            # 计算拟合损失
            loss_fit = F.smooth_l1_loss(deformed_points, normalized_target)

            # 计算正则化损失（平滑约束）
            loss_reg = self._compute_regularization_loss()

            # 总损失
            loss = loss_fit + lambda_reg * loss_reg
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Step {i}: Fit Loss = {loss_fit.item():.6e}, "
                      f"Reg Loss = {loss_reg.item():.6e}, Total Loss = {loss.item():.6e}")

        # 将形变参数设为不可学习（推理模式）
        self.cage_offsets = self.cage_offsets.detach()

        # 返回形变后的点云（世界坐标系）
        final_points = self._apply_deformation(aligned_source)

        print("Optimization Done.")
        return final_points

    def _compute_regularization_loss(self) -> torch.Tensor:
        """
        计算平滑约束 (Laplacian Smoothness)。
        确保体素网格的变形是平滑的。

        Returns:
            正则化损失
        """
        grid = self.cage_offsets

        # 计算二阶差分 (Laplacian近似)
        dzz = grid[:, :, 2:, :, :] - 2 * grid[:, :, 1:-1, :, :] + grid[:, :, :-2, :, :]
        dyy = grid[:, :, :, 2:, :] - 2 * grid[:, :, :, 1:-1, :] + grid[:, :, :, :-2, :]
        dxx = grid[:, :, :, :, 2:] - 2 * grid[:, :, :, :, 1:-1] + grid[:, :, :, :, :-2]

        loss_smooth = torch.mean(dzz**2) + torch.mean(dyy**2) + torch.mean(dxx**2)
        loss_magnitude = torch.mean(grid**2)

        return loss_smooth + 0.01 * loss_magnitude

    def _apply_deformation(self, points: torch.Tensor) -> torch.Tensor:
        """
        对点云应用已学习的形变。

        Args:
            points: 世界坐标系点云 (N, 3)

        Returns:
            形变后的点云 (N, 3)
        """
        norm_coords = self._normalize_coords(points)

        point_offsets = F.grid_sample(
            self.cage_offsets,
            norm_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        point_offsets = point_offsets.view(3, -1).permute(1, 0)

        # 归一化空间中的点
        normalized_points = (points - self.center) / self.scale
        deformed_normalized = normalized_points + point_offsets

        # 反归一化回世界坐标系
        deformed_world = deformed_normalized * self.scale + self.center

        return deformed_world

    def queryPoints(
        self,
        world_points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        """
        查询原始空间中的点云 world_points 在按照体素的顶点插值的坐标在形变后的体素场中的位置。

        需要先调用 loadPoints 和 deformPoints。

        Args:
            world_points: 原始空间中的点云（世界坐标系）(N, 3)

        Returns:
            形变后的点云 (N, 3)
        """
        if self.cage_offsets is None:
            raise RuntimeError("请先调用 loadPoints 加载点云并创建体素网格！")

        world_points = toTensor(world_points, self.dtype, self.device)

        # 应用已学习的形变
        deformed_points = self._apply_deformation(world_points)

        return deformed_points

    def getVoxelVertices(self) -> torch.Tensor:
        """
        获取体素网格的顶点坐标（世界坐标系）。

        Returns:
            体素顶点坐标 ((W+1)*(H+1)*(D+1), 3)
        """
        if self.min_coords is None:
            raise RuntimeError("请先调用 loadPoints 加载点云并创建体素网格！")

        # 生成体素网格顶点的归一化坐标
        x = torch.linspace(self.min_coords[0].item(), self.max_coords[0].item(), self.grid_W + 1, device=self.device)
        y = torch.linspace(self.min_coords[1].item(), self.max_coords[1].item(), self.grid_H + 1, device=self.device)
        z = torch.linspace(self.min_coords[2].item(), self.max_coords[2].item(), self.grid_D + 1, device=self.device)

        # 创建网格
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        vertices_normalized = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        # 转换到世界坐标系
        vertices_world = vertices_normalized * self.scale + self.center

        return vertices_world

    def getDeformedVoxelVertices(self) -> torch.Tensor:
        """
        获取形变后的体素网格顶点坐标（世界坐标系）。

        Returns:
            形变后的体素顶点坐标 ((W+1)*(H+1)*(D+1), 3)
        """
        if self.cage_offsets is None:
            raise RuntimeError("请先调用 loadPoints 和 deformPoints！")

        # 获取原始顶点
        vertices_world = self.getVoxelVertices()

        # 应用形变
        deformed_vertices = self._apply_deformation(vertices_world)

        return deformed_vertices
