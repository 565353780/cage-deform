import torch
import numpy as np
from typing import Union, Optional, Literal

from cage_deform.Method.data import toTensor


class RBFDeformer(object):
    """
    基于径向基函数 (Radial Basis Functions, RBF) 的空间变形器。
    
    RBF 变形是一种全局插值方法，通过在控制点处定义变形并使用
    径向基函数进行全空间的平滑插值。
    
    优点：
    - 不需要构建 Cage/网格结构
    - 使用适当的核函数可以获得 C^∞ 光滑度
    - 对散乱控制点的支持更好
    
    支持的核函数：
    - 'gaussian': exp(-r²/σ²) - C^∞ 光滑
    - 'thin_plate': r²log(r) - C^1 光滑（2D 中 C^∞）
    - 'triharmonic': r³ - C^1 光滑，适合 3D
    - 'multiquadric': sqrt(r² + σ²) - C^∞ 光滑
    - 'inverse_multiquadric': 1/sqrt(r² + σ²) - C^∞ 光滑
    """
    
    def __init__(
        self,
        kernel: Literal['gaussian', 'thin_plate', 'triharmonic', 
                       'multiquadric', 'inverse_multiquadric'] = 'triharmonic',
        sigma: float = 1.0,
        regularization: float = 1e-4,
        dtype=torch.float32,
        device: str = 'cpu',
    ) -> None:
        """
        初始化 RBF 变形器。
        
        Args:
            kernel: 核函数类型
            sigma: 核函数的尺度参数（用于 gaussian, multiquadric 等）
            regularization: 正则化参数，防止矩阵奇异
            dtype: 数据类型
            device: 计算设备
        """
        self.kernel = kernel
        self.sigma = sigma
        self.regularization = regularization
        self.dtype = dtype
        self.device = device
        
        # 控制点和权重
        self.control_points: Optional[torch.Tensor] = None  # (M, 3)
        self.weights: Optional[torch.Tensor] = None  # (M+4, 3) 包含仿射项
        
        # 归一化参数
        self.center: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        return

    def _compute_kernel_matrix(
        self, 
        points1: torch.Tensor, 
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算两组点之间的核矩阵。
        
        Args:
            points1: 第一组点 (N, 3)
            points2: 第二组点 (M, 3)
            
        Returns:
            核矩阵 (N, M)
        """
        # 计算距离矩阵
        diff = points1.unsqueeze(1) - points2.unsqueeze(0)  # (N, M, 3)
        r2 = torch.sum(diff ** 2, dim=-1)  # (N, M)
        r = torch.sqrt(r2 + 1e-10)  # 防止 sqrt(0)
        
        if self.kernel == 'gaussian':
            # φ(r) = exp(-r²/σ²)
            K = torch.exp(-r2 / (self.sigma ** 2))
            
        elif self.kernel == 'thin_plate':
            # φ(r) = r²log(r)，r=0 时为 0
            K = torch.where(
                r > 1e-10,
                r2 * torch.log(r),
                torch.zeros_like(r2)
            )
            
        elif self.kernel == 'triharmonic':
            # φ(r) = r³
            K = r ** 3
            
        elif self.kernel == 'multiquadric':
            # φ(r) = sqrt(r² + σ²)
            K = torch.sqrt(r2 + self.sigma ** 2)
            
        elif self.kernel == 'inverse_multiquadric':
            # φ(r) = 1/sqrt(r² + σ²)
            K = 1.0 / torch.sqrt(r2 + self.sigma ** 2)
            
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        return K

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

    def fit(
        self,
        source_points: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        use_affine_init: bool = True,
    ) -> 'RBFDeformer':
        """
        拟合 RBF 变形场。
        
        给定源点和目标点，计算 RBF 权重使得变形后的源点与目标点对齐。
        变形公式：f(x) = Σ w_i φ(||x - c_i||) + Ax + b
        
        其中：
        - φ 是径向基函数
        - c_i 是控制点（源点）
        - w_i 是权重
        - Ax + b 是仿射项（保证多项式精确性）

        Args:
            source_points: 源控制点（世界坐标系）(M, 3)
            target_points: 目标控制点（世界坐标系）(M, 3)
            use_affine_init: 是否先用仿射变换初步对齐

        Returns:
            self
        """
        source_points = toTensor(source_points, self.dtype, self.device)
        target_points = toTensor(target_points, self.dtype, self.device)
        
        M = source_points.shape[0]
        print(f"Fitting RBF with {M} control points, kernel={self.kernel}")
        
        # 计算归一化参数
        all_points = torch.cat([source_points, target_points], dim=0)
        min_coords = all_points.min(dim=0)[0]
        max_coords = all_points.max(dim=0)[0]
        self.center = (min_coords + max_coords) / 2.0
        self.scale = (max_coords - min_coords).max() / 2.0
        
        # 归一化
        norm_source = (source_points - self.center) / self.scale
        norm_target = (target_points - self.center) / self.scale
        
        # 可选：先用仿射变换初步对齐
        if use_affine_init:
            affine = self._compute_affine_transform(norm_source, norm_target)
            source_np = norm_source.cpu().numpy()
            source_h = np.concatenate([source_np, np.ones((M, 1), dtype=np.float32)], axis=1)
            aligned_source_np = (source_h @ affine.T)[:, :3]
            norm_source = toTensor(aligned_source_np, self.dtype, self.device)
        
        self.control_points = norm_source.clone()
        
        # 计算位移（目标 - 源）
        displacements = norm_target - norm_source  # (M, 3)
        
        # 构建线性系统：[K P; P^T 0] [w; a] = [d; 0]
        # K: 核矩阵 (M, M)
        # P: 多项式矩阵 [1, x, y, z] (M, 4)
        # w: RBF 权重 (M, 3)
        # a: 仿射系数 (4, 3)
        
        K = self._compute_kernel_matrix(norm_source, norm_source)  # (M, M)
        
        # 添加正则化
        K = K + self.regularization * torch.eye(M, dtype=self.dtype, device=self.device)
        
        # 多项式矩阵 P = [1, x, y, z]
        P = torch.cat([
            torch.ones(M, 1, dtype=self.dtype, device=self.device),
            norm_source
        ], dim=1)  # (M, 4)
        
        # 构建扩展系统矩阵
        # [K  P ] [w]   [d]
        # [P' 0 ] [a] = [0]
        zeros_44 = torch.zeros(4, 4, dtype=self.dtype, device=self.device)
        zeros_43 = torch.zeros(4, 3, dtype=self.dtype, device=self.device)
        
        A_top = torch.cat([K, P], dim=1)  # (M, M+4)
        A_bottom = torch.cat([P.T, zeros_44], dim=1)  # (4, M+4)
        A = torch.cat([A_top, A_bottom], dim=0)  # (M+4, M+4)
        
        b = torch.cat([displacements, zeros_43], dim=0)  # (M+4, 3)
        
        # 求解线性系统
        try:
            self.weights = torch.linalg.solve(A, b)  # (M+4, 3)
        except Exception as e:
            print(f"Warning: solve failed, using lstsq. Error: {e}")
            self.weights = torch.linalg.lstsq(A, b).solution
        
        print(f"RBF fitting done. Weight shape: {self.weights.shape}")
        return self

    def transform(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        """
        使用已拟合的 RBF 变形场变换点云。
        
        Args:
            points: 输入点云（世界坐标系）(N, 3)
            
        Returns:
            变形后的点云 (N, 3)
        """
        if self.weights is None:
            raise RuntimeError("请先调用 fit() 拟合 RBF 变形场！")
        
        points = toTensor(points, self.dtype, self.device)
        
        # 归一化
        norm_points = (points - self.center) / self.scale
        
        # 计算核值
        K = self._compute_kernel_matrix(norm_points, self.control_points)  # (N, M)
        
        # 多项式项
        P = torch.cat([
            torch.ones(norm_points.shape[0], 1, dtype=self.dtype, device=self.device),
            norm_points
        ], dim=1)  # (N, 4)
        
        M = self.control_points.shape[0]
        w_rbf = self.weights[:M]  # (M, 3)
        w_affine = self.weights[M:]  # (4, 3)
        
        # 计算位移：Kw + Pa
        displacement = K @ w_rbf + P @ w_affine  # (N, 3)
        
        # 应用位移
        deformed_norm = norm_points + displacement
        
        # 反归一化
        deformed_world = deformed_norm * self.scale + self.center
        
        return deformed_world

    def deformPoints(
        self,
        source_points: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        query_points: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
    ) -> torch.Tensor:
        """
        便捷方法：拟合并变换点云。
        
        Args:
            source_points: 源控制点（世界坐标系）(M, 3)
            target_points: 目标控制点（世界坐标系）(M, 3)
            query_points: 要变换的点云，如果为 None 则变换 source_points
            
        Returns:
            变形后的点云
        """
        self.fit(source_points, target_points)
        
        if query_points is None:
            query_points = source_points
        
        return self.transform(query_points)

    def queryPoints(
        self,
        world_points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        """
        查询点云在形变后空间中的位置（transform 的别名）。

        Args:
            world_points: 原始空间中的点云（世界坐标系）(N, 3)

        Returns:
            形变后的点云 (N, 3)
        """
        return self.transform(world_points)

    def get_jacobian(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        """
        计算变形场在给定点处的雅可比矩阵。
        
        雅可比矩阵描述了变形的局部线性近似，可用于分析变形的性质
        （如是否保持方向、局部放大/缩小等）。
        
        Args:
            points: 查询点（世界坐标系）(N, 3)
            
        Returns:
            雅可比矩阵 (N, 3, 3)
        """
        if self.weights is None:
            raise RuntimeError("请先调用 fit() 拟合 RBF 变形场！")
        
        points = toTensor(points, self.dtype, self.device)
        N = points.shape[0]
        M = self.control_points.shape[0]
        
        # 归一化
        norm_points = (points - self.center) / self.scale
        
        # 计算距离
        diff = norm_points.unsqueeze(1) - self.control_points.unsqueeze(0)  # (N, M, 3)
        r2 = torch.sum(diff ** 2, dim=-1)  # (N, M)
        r = torch.sqrt(r2 + 1e-10)
        
        # 计算核函数导数
        if self.kernel == 'gaussian':
            # dφ/dr = -2r/σ² * exp(-r²/σ²)
            dphi_dr = -2 * r / (self.sigma ** 2) * torch.exp(-r2 / (self.sigma ** 2))
        elif self.kernel == 'thin_plate':
            # dφ/dr = 2r*log(r) + r = r(2log(r) + 1)
            dphi_dr = torch.where(
                r > 1e-10,
                r * (2 * torch.log(r) + 1),
                torch.zeros_like(r)
            )
        elif self.kernel == 'triharmonic':
            # dφ/dr = 3r²
            dphi_dr = 3 * r2
        elif self.kernel == 'multiquadric':
            # dφ/dr = r/sqrt(r² + σ²)
            dphi_dr = r / torch.sqrt(r2 + self.sigma ** 2)
        elif self.kernel == 'inverse_multiquadric':
            # dφ/dr = -r/(r² + σ²)^(3/2)
            dphi_dr = -r / (r2 + self.sigma ** 2) ** 1.5
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        # dr/dx_j = (x_j - c_ij) / r
        dr_dx = diff / (r.unsqueeze(-1) + 1e-10)  # (N, M, 3)
        
        # dφ/dx_j = dφ/dr * dr/dx_j
        dphi_dx = dphi_dr.unsqueeze(-1) * dr_dx  # (N, M, 3)
        
        w_rbf = self.weights[:M]  # (M, 3)
        w_affine = self.weights[M:]  # (4, 3) -> [b; A] where A is (3, 3)
        
        # RBF 项的雅可比：Σ w_i ⊗ dφ_i/dx
        # J_rbf[k, j] = Σ_i w_i[k] * dφ_i/dx[j]
        J_rbf = torch.einsum('nmi,mk->nmk', dphi_dx, w_rbf)  # (N, M, 3)
        J_rbf = J_rbf.sum(dim=1)  # (N, 3)... wait this is wrong
        
        # 重新计算：对于每个输出维度 k，J[n,k,j] = Σ_i w_rbf[i,k] * dphi_dx[n,i,j]
        # 即 J = dphi_dx @ w_rbf，但需要正确的 einsum
        # dphi_dx: (N, M, 3), w_rbf: (M, 3)
        # J_rbf[n, k, j] = Σ_i dphi_dx[n, i, j] * w_rbf[i, k]
        J_rbf = torch.einsum('nmj,mk->nkj', dphi_dx, w_rbf)  # (N, 3, 3)
        
        # 仿射项的雅可比：A^T（因为 Pa 中 P = [1, x, y, z]）
        # w_affine 的形状是 (4, 3)，第一行是常数项，后三行是线性项
        A = w_affine[1:, :]  # (3, 3)
        J_affine = A.T.unsqueeze(0).expand(N, -1, -1)  # (N, 3, 3)
        
        # 总雅可比 = I + J_rbf + J_affine
        I = torch.eye(3, dtype=self.dtype, device=self.device).unsqueeze(0).expand(N, -1, -1)
        J = I + J_rbf + J_affine
        
        return J

    def save(self, path: str) -> None:
        """保存 RBF 模型到文件。"""
        state = {
            'kernel': self.kernel,
            'sigma': self.sigma,
            'regularization': self.regularization,
            'control_points': self.control_points.cpu().numpy() if self.control_points is not None else None,
            'weights': self.weights.cpu().numpy() if self.weights is not None else None,
            'center': self.center.cpu().numpy() if self.center is not None else None,
            'scale': self.scale.item() if self.scale is not None else None,
        }
        np.savez(path, **state)
        print(f"RBF model saved to {path}")

    def load(self, path: str) -> 'RBFDeformer':
        """从文件加载 RBF 模型。"""
        data = np.load(path, allow_pickle=True)
        self.kernel = str(data['kernel'])
        self.sigma = float(data['sigma'])
        self.regularization = float(data['regularization'])
        
        if data['control_points'] is not None:
            self.control_points = toTensor(data['control_points'], self.dtype, self.device)
        if data['weights'] is not None:
            self.weights = toTensor(data['weights'], self.dtype, self.device)
        if data['center'] is not None:
            self.center = toTensor(data['center'], self.dtype, self.device)
        if data['scale'] is not None:
            self.scale = torch.tensor(data['scale'], dtype=self.dtype, device=self.device)
        
        print(f"RBF model loaded from {path}")
        return self
