import torch
import numpy as np
from typing import Union, Optional, Literal

from cage_deform.Method.data import toTensor


class MLSDeformer(object):
    """
    基于移动最小二乘 (Moving Least Squares, MLS) 的空间变形器。
    
    MLS 是一种局部加权插值方法，对于每个查询点，根据其与控制点的距离
    计算一个局部的最优变换。
    
    支持三种变换模式：
    - 'affine': 仿射变换（最灵活，可能导致剪切）
    - 'similarity': 相似变换（保持角度，允许均匀缩放和旋转）
    - 'rigid': 刚性变换（只允许旋转和平移，保持形状）
    
    优点：
    - 自然保持局部几何特性
    - 刚性模式可以很好地保持曲率
    - 不需要求解全局线性系统
    
    参考：
    - Schaefer et al., "Image Deformation Using Moving Least Squares", 2006
    """
    
    def __init__(
        self,
        mode: Literal['affine', 'similarity', 'rigid'] = 'rigid',
        alpha: float = 2.0,
        dtype=torch.float32,
        device: str = 'cpu',
    ) -> None:
        """
        初始化 MLS 变形器。
        
        Args:
            mode: 变换模式 ('affine', 'similarity', 'rigid')
            alpha: 权重函数的指数，控制影响范围的衰减速度
                   较大的 alpha 使变形更局部化
            dtype: 数据类型
            device: 计算设备
        """
        self.mode = mode
        self.alpha = alpha
        self.dtype = dtype
        self.device = device
        
        # 控制点
        self.source_points: Optional[torch.Tensor] = None  # (M, 3)
        self.target_points: Optional[torch.Tensor] = None  # (M, 3)
        
        # 归一化参数
        self.center: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        
        # 预计算的仿射初始化变换
        self.affine_init: Optional[np.ndarray] = None
        return

    def _compute_weights(
        self, 
        query_points: torch.Tensor,
        control_points: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        计算查询点相对于控制点的权重。
        
        权重函数: w_i(x) = 1 / ||x - p_i||^(2*alpha)
        
        Args:
            query_points: 查询点 (N, 3)
            control_points: 控制点 (M, 3)
            eps: 防止除零的小量
            
        Returns:
            权重矩阵 (N, M)
        """
        # 计算距离的平方
        diff = query_points.unsqueeze(1) - control_points.unsqueeze(0)  # (N, M, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (N, M)
        
        # 权重: w = 1 / d^(2*alpha)
        weights = 1.0 / (dist_sq ** self.alpha + eps)
        
        return weights

    def _compute_affine_transform(
        self,
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
    ) -> np.ndarray:
        """
        计算全局仿射变换矩阵。
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
    ) -> 'MLSDeformer':
        """
        设置 MLS 变形的控制点。
        
        注意：MLS 不需要预先求解任何方程，变形是在查询时实时计算的。
        这里只是存储控制点并进行可选的仿射预对齐。

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
        print(f"Setting up MLS with {M} control points, mode={self.mode}")
        
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
            self.affine_init = self._compute_affine_transform(norm_source, norm_target)
            source_np = norm_source.cpu().numpy()
            source_h = np.concatenate([source_np, np.ones((M, 1), dtype=np.float32)], axis=1)
            aligned_source_np = (source_h @ self.affine_init.T)[:, :3]
            norm_source = toTensor(aligned_source_np, self.dtype, self.device)
        
        self.source_points = norm_source
        self.target_points = norm_target
        
        return self

    def _transform_affine(
        self,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用仿射 MLS 变换点云。
        
        对于每个查询点 v，计算加权仿射变换：
        f(v) = (v - p*) @ M + q*
        
        其中：
        - p* = Σ w_i p_i / Σ w_i （加权质心）
        - q* = Σ w_i q_i / Σ w_i
        - M 是最小化 Σ w_i ||p̂_i M - q̂_i||² 的矩阵
        """
        N = query_points.shape[0]
        M_ctrl = self.source_points.shape[0]
        
        # 计算权重 (N, M)
        weights = self._compute_weights(query_points, self.source_points)
        weights_sum = weights.sum(dim=1, keepdim=True)  # (N, 1)
        weights_normalized = weights / weights_sum  # (N, M)
        
        # 加权质心
        p_star = (weights_normalized @ self.source_points)  # (N, 3)
        q_star = (weights_normalized @ self.target_points)  # (N, 3)
        
        # 中心化的控制点
        p_hat = self.source_points.unsqueeze(0) - p_star.unsqueeze(1)  # (N, M, 3)
        q_hat = self.target_points.unsqueeze(0) - q_star.unsqueeze(1)  # (N, M, 3)
        
        # 计算仿射矩阵 M（对每个查询点）
        # M = (Σ w_i p̂_i^T p̂_i)^(-1) (Σ w_i p̂_i^T q̂_i)
        # 使用 einsum 批量计算
        
        # p̂^T p̂ 加权求和: (N, 3, 3)
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (N, M, 1, 1)
        p_hat_outer = torch.einsum('nmi,nmj->nmij', p_hat, p_hat)  # (N, M, 3, 3)
        A = (weights_expanded * p_hat_outer).sum(dim=1)  # (N, 3, 3)
        
        # p̂^T q̂ 加权求和: (N, 3, 3)
        pq_outer = torch.einsum('nmi,nmj->nmij', p_hat, q_hat)  # (N, M, 3, 3)
        B = (weights_expanded * pq_outer).sum(dim=1)  # (N, 3, 3)
        
        # 求解 M = A^(-1) B
        # 添加正则化防止奇异
        reg = 1e-6 * torch.eye(3, dtype=self.dtype, device=self.device)
        A = A + reg.unsqueeze(0)
        
        try:
            M_transform = torch.linalg.solve(A, B)  # (N, 3, 3)
        except:
            M_transform = torch.linalg.lstsq(A, B).solution
        
        # 应用变换
        v_centered = query_points - p_star  # (N, 3)
        result = torch.einsum('ni,nij->nj', v_centered, M_transform) + q_star  # (N, 3)
        
        return result

    def _transform_similarity(
        self,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用相似 MLS 变换点云。
        
        相似变换保持角度，允许均匀缩放和旋转。
        """
        N = query_points.shape[0]
        
        # 计算权重
        weights = self._compute_weights(query_points, self.source_points)
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights_normalized = weights / weights_sum
        
        # 加权质心
        p_star = weights_normalized @ self.source_points  # (N, 3)
        q_star = weights_normalized @ self.target_points  # (N, 3)
        
        # 中心化
        p_hat = self.source_points.unsqueeze(0) - p_star.unsqueeze(1)  # (N, M, 3)
        q_hat = self.target_points.unsqueeze(0) - q_star.unsqueeze(1)  # (N, M, 3)
        
        # 计算缩放因子 μ
        weights_expanded = weights.unsqueeze(-1)  # (N, M, 1)
        mu = (weights_expanded * (p_hat ** 2)).sum(dim=(1, 2))  # (N,)
        
        # 计算旋转矩阵（使用 SVD）
        # 先计算 Σ w_i p̂_i^T q̂_i
        H = torch.einsum('nm,nmi,nmj->nij', weights, p_hat, q_hat)  # (N, 3, 3)
        
        # SVD 分解
        U, S, Vh = torch.linalg.svd(H)
        
        # R = V U^T
        R = Vh.transpose(-2, -1) @ U.transpose(-2, -1)  # (N, 3, 3)
        
        # 检测反射（行列式为负）
        det = torch.linalg.det(R)  # (N,)
        
        # 修正反射
        mask = det < 0
        if mask.any():
            # 翻转 V 的最后一列
            Vh_corrected = Vh.clone()
            Vh_corrected[mask, :, -1] *= -1
            R[mask] = Vh_corrected[mask].transpose(-2, -1) @ U[mask].transpose(-2, -1)
        
        # 应用变换
        v_centered = query_points - p_star  # (N, 3)
        result = torch.einsum('ni,nij->nj', v_centered, R) + q_star
        
        return result

    def _transform_rigid(
        self,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用刚性 MLS 变换点云。
        
        刚性变换只允许旋转和平移，保持距离不变。
        这是保持局部几何特性最好的模式。
        """
        N = query_points.shape[0]
        
        # 计算权重
        weights = self._compute_weights(query_points, self.source_points)
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights_normalized = weights / weights_sum
        
        # 加权质心
        p_star = weights_normalized @ self.source_points  # (N, 3)
        q_star = weights_normalized @ self.target_points  # (N, 3)
        
        # 中心化
        p_hat = self.source_points.unsqueeze(0) - p_star.unsqueeze(1)  # (N, M, 3)
        q_hat = self.target_points.unsqueeze(0) - q_star.unsqueeze(1)  # (N, M, 3)
        
        # 计算协方差矩阵 H = Σ w_i p̂_i^T q̂_i
        H = torch.einsum('nm,nmi,nmj->nij', weights, p_hat, q_hat)  # (N, 3, 3)
        
        # 使用 SVD 计算最优旋转
        U, S, Vh = torch.linalg.svd(H)
        
        # R = V U^T（Kabsch 算法）
        R = Vh.transpose(-2, -1) @ U.transpose(-2, -1)  # (N, 3, 3)
        
        # 检测并修正反射
        det = torch.linalg.det(R)
        mask = det < 0
        if mask.any():
            # 翻转 V 的最后一列
            Vh_corrected = Vh.clone()
            Vh_corrected[mask, :, -1] *= -1
            R[mask] = Vh_corrected[mask].transpose(-2, -1) @ U[mask].transpose(-2, -1)
        
        # 应用刚性变换: f(v) = (v - p*) R + q*
        v_centered = query_points - p_star  # (N, 3)
        result = torch.einsum('ni,nij->nj', v_centered, R) + q_star
        
        return result

    def transform(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
    ) -> torch.Tensor:
        """
        使用 MLS 变形场变换点云。
        
        Args:
            points: 输入点云（世界坐标系）(N, 3)
            
        Returns:
            变形后的点云 (N, 3)
        """
        if self.source_points is None:
            raise RuntimeError("请先调用 fit() 设置控制点！")
        
        points = toTensor(points, self.dtype, self.device)
        
        # 归一化
        norm_points = (points - self.center) / self.scale
        
        # 可选：应用仿射预对齐
        if self.affine_init is not None:
            points_np = norm_points.cpu().numpy()
            points_h = np.concatenate([points_np, np.ones((points_np.shape[0], 1), dtype=np.float32)], axis=1)
            aligned_np = (points_h @ self.affine_init.T)[:, :3]
            norm_points = toTensor(aligned_np, self.dtype, self.device)
        
        # 根据模式选择变换方法
        if self.mode == 'affine':
            result_norm = self._transform_affine(norm_points)
        elif self.mode == 'similarity':
            result_norm = self._transform_similarity(norm_points)
        elif self.mode == 'rigid':
            result_norm = self._transform_rigid(norm_points)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # 反归一化
        result_world = result_norm * self.scale + self.center
        
        return result_world

    def deformPoints(
        self,
        source_points: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        query_points: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
    ) -> torch.Tensor:
        """
        便捷方法：设置控制点并变换点云。
        
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

    def get_local_transform(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
    ) -> tuple:
        """
        获取每个查询点处的局部变换参数。
        
        对于刚性模式，返回旋转矩阵和平移向量。
        
        Args:
            points: 查询点（世界坐标系）(N, 3)
            
        Returns:
            (rotations, translations): 
                - rotations: 旋转矩阵 (N, 3, 3)
                - translations: 平移向量 (N, 3)
        """
        if self.source_points is None:
            raise RuntimeError("请先调用 fit() 设置控制点！")
        
        points = toTensor(points, self.dtype, self.device)
        N = points.shape[0]
        
        # 归一化
        norm_points = (points - self.center) / self.scale
        
        if self.affine_init is not None:
            points_np = norm_points.cpu().numpy()
            points_h = np.concatenate([points_np, np.ones((N, 1), dtype=np.float32)], axis=1)
            aligned_np = (points_h @ self.affine_init.T)[:, :3]
            norm_points = toTensor(aligned_np, self.dtype, self.device)
        
        # 计算权重
        weights = self._compute_weights(norm_points, self.source_points)
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights_normalized = weights / weights_sum
        
        # 加权质心
        p_star = weights_normalized @ self.source_points
        q_star = weights_normalized @ self.target_points
        
        # 中心化
        p_hat = self.source_points.unsqueeze(0) - p_star.unsqueeze(1)
        q_hat = self.target_points.unsqueeze(0) - q_star.unsqueeze(1)
        
        # 协方差矩阵
        H = torch.einsum('nm,nmi,nmj->nij', weights, p_hat, q_hat)
        
        # SVD
        U, S, Vh = torch.linalg.svd(H)
        R = Vh.transpose(-2, -1) @ U.transpose(-2, -1)
        
        # 修正反射
        det = torch.linalg.det(R)
        mask = det < 0
        if mask.any():
            Vh_corrected = Vh.clone()
            Vh_corrected[mask, :, -1] *= -1
            R[mask] = Vh_corrected[mask].transpose(-2, -1) @ U[mask].transpose(-2, -1)
        
        # 平移：t = q* - p* @ R
        translations = q_star - torch.einsum('ni,nij->nj', p_star, R)
        
        return R, translations

    def save(self, path: str) -> None:
        """保存 MLS 模型到文件。"""
        state = {
            'mode': self.mode,
            'alpha': self.alpha,
            'source_points': self.source_points.cpu().numpy() if self.source_points is not None else None,
            'target_points': self.target_points.cpu().numpy() if self.target_points is not None else None,
            'center': self.center.cpu().numpy() if self.center is not None else None,
            'scale': self.scale.item() if self.scale is not None else None,
            'affine_init': self.affine_init,
        }
        np.savez(path, **state)
        print(f"MLS model saved to {path}")

    def load(self, path: str) -> 'MLSDeformer':
        """从文件加载 MLS 模型。"""
        data = np.load(path, allow_pickle=True)
        self.mode = str(data['mode'])
        self.alpha = float(data['alpha'])
        
        if data['source_points'] is not None:
            self.source_points = toTensor(data['source_points'], self.dtype, self.device)
        if data['target_points'] is not None:
            self.target_points = toTensor(data['target_points'], self.dtype, self.device)
        if data['center'] is not None:
            self.center = toTensor(data['center'], self.dtype, self.device)
        if data['scale'] is not None:
            self.scale = torch.tensor(data['scale'], dtype=self.dtype, device=self.device)
        if data['affine_init'] is not None:
            self.affine_init = data['affine_init']
        
        print(f"MLS model loaded from {path}")
        return self
