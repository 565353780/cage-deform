import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from typing import Union, Optional

if torch.cuda.is_available():
    from cage_deform.Lib.chamfer3D.dist_chamfer_3D import chamfer_3DDist
else:
    from cage_deform.Lib.chamfer3D.chamfer_python import distChamfer

from cage_deform.Method.data import toTensor


class BSplineDeformer(object):
    """
    基于三次 B 样条的自由形式变形 (B-Spline FFD)。

    与传统的三线性插值不同，三次 B 样条使用 4×4×4 = 64 个控制点来计算
    每个点的位置，这保证了 C² 连续性（二阶导数连续），从而避免了
    分段仿射变形导致的曲率断裂问题。

    三次 B 样条基函数：
    - B0(t) = (1-t)³/6
    - B1(t) = (3t³ - 6t² + 4)/6  
    - B2(t) = (-3t³ + 3t² + 3t + 1)/6
    - B3(t) = t³/6
    """
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
        # 控制网格尺寸（注意：为了支持边界处的 B 样条插值，需要额外的控制点）
        self.grid_W: int = 0
        self.grid_H: int = 0
        self.grid_D: int = 0

        # 归一化参数
        self.center: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.min_coords: Optional[torch.Tensor] = None
        self.max_coords: Optional[torch.Tensor] = None
        self.extent: Optional[torch.Tensor] = None

        # 控制点偏移量（形变参数）
        # 形状为 (D+3, H+3, W+3, 3)，额外的控制点用于边界处理
        self.control_offsets: Optional[torch.Tensor] = None

        # 初始点云（归一化空间）
        self.initial_points: Optional[torch.Tensor] = None

        if torch.cuda.is_available() and device != "cpu":
            self.chamfer_func = chamfer_3DDist()
        else:
            self.chamfer_func = distChamfer
        return

    @staticmethod
    def _bspline_basis(t: torch.Tensor) -> tuple:
        """
        计算三次 B 样条的 4 个基函数值。

        Args:
            t: 局部坐标 [0, 1) 范围内的值

        Returns:
            (B0, B1, B2, B3): 四个基函数在 t 处的值
        """
        t2 = t * t
        t3 = t2 * t
        one_minus_t = 1.0 - t
        one_minus_t2 = one_minus_t * one_minus_t
        one_minus_t3 = one_minus_t2 * one_minus_t

        # 三次 B 样条基函数（均匀节点）
        B0 = one_minus_t3 / 6.0
        B1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
        B2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
        B3 = t3 / 6.0

        return B0, B1, B2, B3

    @staticmethod
    def _bspline_basis_derivative(t: torch.Tensor) -> tuple:
        """
        计算三次 B 样条基函数的一阶导数。

        Args:
            t: 局部坐标 [0, 1) 范围内的值

        Returns:
            (dB0, dB1, dB2, dB3): 四个基函数导数在 t 处的值
        """
        t2 = t * t
        one_minus_t = 1.0 - t
        one_minus_t2 = one_minus_t * one_minus_t

        dB0 = -one_minus_t2 / 2.0
        dB1 = (3.0 * t2 - 4.0 * t) / 2.0
        dB2 = (-3.0 * t2 + 2.0 * t + 1.0) / 2.0
        dB3 = t2 / 2.0

        return dB0, dB1, dB2, dB3

    def loadPoints(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
        voxel_size: float = 0.125,
        padding: float = 0.1,
    ) -> dict:
        """
        加载点云并创建 B 样条控制网格。

        注意：为了在边界处也能进行完整的 4×4×4 插值，控制网格会在
        每个方向上多出 3 个控制点（前后各 1.5 个体素的额外空间）。

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

        # 计算 center 和 scale
        self.center = (min_coords + max_coords) / 2.0
        self.scale = extent.max() / 2.0

        # 2. 归一化点云到 [-0.5, 0.5]
        normalized_points = (points - self.center) / self.scale
        self.initial_points = normalized_points.clone()

        # 3. 计算带 padding 的边界框
        normalized_min = normalized_points.min(dim=0)[0]
        normalized_max = normalized_points.max(dim=0)[0]
        normalized_extent = normalized_max - normalized_min
        padding_size = normalized_extent * padding

        self.min_coords = normalized_min - padding_size
        self.max_coords = normalized_max + padding_size
        self.extent = self.max_coords - self.min_coords

        # 4. 根据 voxel_size 计算内部网格的维度
        normalized_voxel_size = voxel_size / self.scale
        dims = (self.extent / normalized_voxel_size).ceil().int()

        # 内部网格尺寸
        self.grid_W = max(dims[0].item(), 2)
        self.grid_H = max(dims[1].item(), 2)
        self.grid_D = max(dims[2].item(), 2)

        # 实际的体素尺寸（可能因为 ceil 略有不同）
        self.actual_voxel_size = self.extent / torch.tensor(
            [self.grid_W, self.grid_H, self.grid_D], 
            dtype=self.dtype, device=self.device
        )

        print(f"B-Spline Control Grid Size: {self.grid_W + 3}x{self.grid_H + 3}x{self.grid_D + 3} (WxHxD)")
        print(f"Inner Grid Size: {self.grid_W}x{self.grid_H}x{self.grid_D}")
        print(f"Normalization: center={self.center.cpu().numpy()}, scale={self.scale.item():.6f}")

        # 5. 初始化控制点的偏移量
        # 为了边界处理，控制网格在每个方向上多 3 个点
        # 这样内部的 grid_W × grid_H × grid_D 个体素都能使用完整的 4×4×4 插值
        self.control_offsets = torch.zeros(
            self.grid_D + 3, self.grid_H + 3, self.grid_W + 3, 3,
            dtype=self.dtype, device=self.device
        )

        return {
            'grid_size': (self.grid_W, self.grid_H, self.grid_D),
            'control_grid_size': (self.grid_W + 3, self.grid_H + 3, self.grid_D + 3),
            'num_control_points': (self.grid_W + 3) * (self.grid_H + 3) * (self.grid_D + 3),
            'center': self.center.cpu().numpy(),
            'scale': self.scale.item(),
            'extent': self.extent.cpu().numpy(),
        }

    def _point_to_cell_coords(self, points: torch.Tensor) -> tuple:
        """
        将归一化空间中的点转换为体素网格坐标。

        Args:
            points: 归一化空间中的点 (N, 3)

        Returns:
            (cell_indices, local_coords): 
                - cell_indices: 点所在的体素索引 (N, 3)
                - local_coords: 点在体素内的局部坐标 [0, 1) (N, 3)
        """
        # 归一化到 [0, grid_size] 范围
        grid_coords = (points - self.min_coords) / self.actual_voxel_size

        # 体素索引（向下取整）
        cell_indices = grid_coords.floor().long()

        # 限制在有效范围内 [0, grid_size - 1]
        cell_indices = torch.clamp(
            cell_indices,
            min=torch.zeros(3, dtype=torch.long, device=self.device),
            max=torch.tensor([self.grid_W - 1, self.grid_H - 1, self.grid_D - 1], 
                           dtype=torch.long, device=self.device)
        )

        # 局部坐标 [0, 1)
        local_coords = grid_coords - cell_indices.float()
        local_coords = torch.clamp(local_coords, 0.0, 0.999999)

        return cell_indices, local_coords

    def _evaluate_bspline(
        self, 
        points: torch.Tensor,
        control_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用三次 B 样条在给定点处计算偏移量。

        对于每个点，使用其周围 4×4×4 = 64 个控制点进行插值。

        Args:
            points: 归一化空间中的点 (N, 3)
            control_offsets: 控制点偏移量 (D+3, H+3, W+3, 3)

        Returns:
            点的偏移量 (N, 3)
        """
        cell_indices, local_coords = self._point_to_cell_coords(points)
        N = points.shape[0]

        # 计算三个方向上的 B 样条基函数值
        Bx = self._bspline_basis(local_coords[:, 0])  # 4 个元素，每个 (N,)
        By = self._bspline_basis(local_coords[:, 1])
        Bz = self._bspline_basis(local_coords[:, 2])

        # 初始化输出
        offsets = torch.zeros(N, 3, dtype=self.dtype, device=self.device)

        # 4×4×4 循环计算 B 样条插值
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    # 控制点索引（注意：cell_indices 对应于控制网格中的偏移 1 位置）
                    # 因为控制网格比内部网格大 3，索引从 0 开始对应 cell -1
                    idx_x = cell_indices[:, 0] + k  # [0, grid_W + 2]
                    idx_y = cell_indices[:, 1] + j  # [0, grid_H + 2]
                    idx_z = cell_indices[:, 2] + i  # [0, grid_D + 2]

                    # 获取控制点偏移量
                    ctrl_offset = control_offsets[idx_z, idx_y, idx_x]  # (N, 3)

                    # 计算权重
                    weight = Bx[k] * By[j] * Bz[i]  # (N,)

                    # 累加贡献
                    offsets += weight.unsqueeze(1) * ctrl_offset

        return offsets

    def _compute_affine_transform(
        self,
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
    ) -> np.ndarray:
        """
        计算从源点到目标点的仿射变换矩阵。
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

    def deformPoints(
        self,
        source_points: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        lr: float = 1e-2,
        lambda_smooth: float = 1e3,
        lambda_magnitude: float = 1.0,
        steps: int = 500,
    ) -> torch.Tensor:
        """
        使用三次 B 样条 FFD 将源点变形到目标点位置。

        这个方法会优化控制网格的偏移量，使得变形后的源点尽可能接近目标点，
        同时保持变形的平滑性（C² 连续）。

        Args:
            source_points: 源点云（世界坐标系）(M, 3)
            target_points: 目标点云（世界坐标系）(M, 3)
            lr: 学习率
            lambda_smooth: 平滑正则化权重
            lambda_magnitude: 幅度正则化权重
            steps: 优化步数

        Returns:
            形变后的源点云 (M, 3)
        """
        if self.control_offsets is None:
            raise RuntimeError("请先调用 loadPoints 加载点云并创建控制网格！")

        torch.set_grad_enabled(True)

        source_points = toTensor(source_points, self.dtype, self.device)
        target_points = toTensor(target_points, self.dtype, self.device)

        print(f"Source points: {source_points.shape[0]}, Target points: {target_points.shape[0]}")

        # 1. 使用仿射变换进行初步对齐
        affine = self._compute_affine_transform(source_points, target_points)
        source_np = source_points.cpu().numpy()
        source_h = np.concatenate([source_np, np.ones((source_np.shape[0], 1), dtype=np.float32)], axis=1)
        aligned_source_np = (source_h @ affine.T)[:, :3]
        aligned_source = toTensor(aligned_source_np, self.dtype, self.device)

        # 2. 转换到归一化空间
        normalized_source = (aligned_source - self.center) / self.scale
        normalized_target = (target_points - self.center) / self.scale

        # 3. 初始化控制点偏移量并设置为可学习
        self.control_offsets = torch.zeros(
            self.grid_D + 3, self.grid_H + 3, self.grid_W + 3, 3,
            dtype=self.dtype, device=self.device, requires_grad=True
        )

        optimizer = optim.AdamW([self.control_offsets], lr=lr)

        for i in range(steps):
            optimizer.zero_grad()

            # 使用 B 样条插值计算偏移量
            point_offsets = self._evaluate_bspline(normalized_source, self.control_offsets)

            # 应用偏移量
            deformed_points = normalized_source + point_offsets

            # 计算拟合损失
            loss_fit = F.smooth_l1_loss(deformed_points, normalized_target)

            # 计算正则化损失
            loss_smooth, loss_mag = self._compute_regularization_loss()

            # 总损失
            loss = loss_fit + lambda_smooth * loss_smooth + lambda_magnitude * loss_mag
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Step {i}: Fit={loss_fit.item():.6e}, "
                      f"Smooth={loss_smooth.item():.6e}, Mag={loss_mag.item():.6e}")

        self.control_offsets = self.control_offsets.detach()

        # 返回形变后的点云
        final_points = self._apply_deformation(aligned_source)

        print("B-Spline FFD Optimization Done.")
        return final_points

    def _chamfer_loss(
        self,
        points_a: torch.Tensor,
        points_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算两点云之间的 Chamfer 距离（支持点数不同）。

        Args:
            points_a: (N, 3) 或 (1, N, 3)
            points_b: (M, 3) 或 (1, M, 3)

        Returns:
            chamfer_loss: 标量
        """
        if points_a.dim() == 2:
            points_a = points_a.unsqueeze(0)
        if points_b.dim() == 2:
            points_b = points_b.unsqueeze(0)
        if torch.cuda.is_available() and points_a.is_cuda:
            dist1, dist2, _, _ = self.chamfer_func(points_a, points_b)
            return dist1.mean() + dist2.mean()
        else:
            dist1, dist2, _, _ = self.chamfer_func(points_a, points_b)
            return dist1.mean() + dist2.mean()

    def deformUnmatchedPoints(
        self,
        source_points: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        lr: float = 1e-2,
        lambda_smooth: float = 1e3,
        lambda_magnitude: float = 1.0,
        steps: int = 500,
    ) -> torch.Tensor:
        """
        使用 B 样条 FFD 将源点云变形，最小化与目标点云之间的 Chamfer 距离。

        允许 source_points 与 target_points 数量不一致（未匹配点云）。
        通过优化控制网格偏移，使变形后的源点云在 Chamfer 意义下逼近目标点云。

        Args:
            source_points: 源点云（世界坐标系）(N, 3)
            target_points: 目标点云（世界坐标系）(M, 3)，M 可与 N 不同
            lr: 学习率
            lambda_smooth: 平滑正则化权重
            lambda_magnitude: 幅度正则化权重
            steps: 优化步数

        Returns:
            形变后的源点云 (N, 3)
        """
        if self.control_offsets is None:
            raise RuntimeError("请先调用 loadPoints 加载点云并创建控制网格！")

        torch.set_grad_enabled(True)

        source_points = toTensor(source_points, self.dtype, self.device)
        target_points = toTensor(target_points, self.dtype, self.device)

        print(f"deformUnmatchedPoints: source {source_points.shape[0]}, target {target_points.shape[0]}")

        # 用 loadPoints 的归一化（source 的 center/scale）将 source 与 target 转到同一归一化空间
        normalized_source = (source_points - self.center) / self.scale
        normalized_target = (target_points - self.center) / self.scale

        # 3. 控制点偏移设为可学习
        self.control_offsets = torch.zeros(
            self.grid_D + 3, self.grid_H + 3, self.grid_W + 3, 3,
            dtype=self.dtype, device=self.device, requires_grad=True
        )

        optimizer = optim.AdamW([self.control_offsets], lr=lr)

        for i in range(steps):
            optimizer.zero_grad()

            point_offsets = self._evaluate_bspline(normalized_source, self.control_offsets)
            deformed_points = normalized_source + point_offsets

            # Chamfer loss（支持 N != M）
            loss_chamfer = self._chamfer_loss(deformed_points, normalized_target)

            loss_smooth, loss_mag = self._compute_regularization_loss()
            loss = loss_chamfer + lambda_smooth * loss_smooth + lambda_magnitude * loss_mag
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Step {i}: Chamfer={loss_chamfer.item():.6e}, "
                      f"Smooth={loss_smooth.item():.6e}, Mag={loss_mag.item():.6e}")

        self.control_offsets = self.control_offsets.detach()

        final_points = self._apply_deformation(source_points)
        print("deformUnmatchedPoints (Chamfer) Optimization Done.")
        return final_points

    def _compute_regularization_loss(self) -> tuple:
        """
        计算正则化损失。

        包括：
        1. 二阶平滑约束（Laplacian）- 确保变形场 C² 平滑
        2. 幅度约束 - 防止控制点偏移过大

        Returns:
            (loss_smooth, loss_magnitude)
        """
        grid = self.control_offsets  # (D+3, H+3, W+3, 3)

        # 二阶差分（Laplacian 近似）
        # 沿 D 轴 (Z)
        dzz = grid[2:, :, :, :] - 2 * grid[1:-1, :, :, :] + grid[:-2, :, :, :]
        # 沿 H 轴 (Y)
        dyy = grid[:, 2:, :, :] - 2 * grid[:, 1:-1, :, :] + grid[:, :-2, :, :]
        # 沿 W 轴 (X)
        dxx = grid[:, :, 2:, :] - 2 * grid[:, :, 1:-1, :] + grid[:, :, :-2, :]

        loss_smooth = torch.mean(dzz**2) + torch.mean(dyy**2) + torch.mean(dxx**2)
        loss_magnitude = torch.mean(grid**2)

        return loss_smooth, loss_magnitude

    def _apply_deformation(self, points: torch.Tensor) -> torch.Tensor:
        """
        对点云应用已学习的 B 样条形变。

        Args:
            points: 世界坐标系点云 (N, 3)。应与 loadPoints 使用同一套世界坐标（即 source）

        Returns:
            形变后的点云 (N, 3)
        """
        normalized_points = (points - self.center) / self.scale
        point_offsets = self._evaluate_bspline(normalized_points, self.control_offsets)
        deformed_normalized = normalized_points + point_offsets
        deformed_world = deformed_normalized * self.scale + self.center
        return deformed_world

    def queryPoints(
        self,
        world_points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        """
        查询点云在形变后空间中的位置。

        Args:
            world_points: 原始空间中的点云（世界坐标系）(N, 3)

        Returns:
            形变后的点云 (N, 3)
        """
        if self.control_offsets is None:
            raise RuntimeError("请先调用 loadPoints 和 deformPoints！")

        world_points = toTensor(world_points, self.dtype, self.device)
        return self._apply_deformation(world_points)

    def getControlPoints(self) -> torch.Tensor:
        """
        获取控制点坐标（世界坐标系）。

        Returns:
            控制点坐标 ((D+3)*(H+3)*(W+3), 3)
        """
        if self.min_coords is None:
            raise RuntimeError("请先调用 loadPoints！")

        # 控制网格范围比内部网格大，需要向外扩展
        # 控制网格从 min_coords - voxel_size 开始
        ctrl_min = self.min_coords - self.actual_voxel_size
        ctrl_max = self.max_coords + 2 * self.actual_voxel_size

        x = torch.linspace(ctrl_min[0].item(), ctrl_max[0].item(), 
                          self.grid_W + 3, device=self.device)
        y = torch.linspace(ctrl_min[1].item(), ctrl_max[1].item(), 
                          self.grid_H + 3, device=self.device)
        z = torch.linspace(ctrl_min[2].item(), ctrl_max[2].item(), 
                          self.grid_D + 3, device=self.device)

        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        vertices_normalized = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        vertices_world = vertices_normalized * self.scale + self.center

        return vertices_world

    def getDeformedControlPoints(self) -> torch.Tensor:
        """
        获取形变后的控制点坐标（世界坐标系）。

        Returns:
            形变后的控制点坐标 ((D+3)*(H+3)*(W+3), 3)
        """
        if self.control_offsets is None:
            raise RuntimeError("请先调用 loadPoints 和 deformPoints！")

        vertices_world = self.getControlPoints()

        # 直接添加控制点偏移（已经在归一化空间中）
        offsets_flat = self.control_offsets.reshape(-1, 3) * self.scale
        deformed_vertices = vertices_world + offsets_flat

        return deformed_vertices
