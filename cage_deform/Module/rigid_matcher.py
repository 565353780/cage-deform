import os
import torch
import numpy as np
import open3d as o3d

from tqdm import tqdm
from typing import Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from cage_deform.Method.pcd import toPcd
from cage_deform.Method.data import toNumpy
from cage_deform.Method.sample import sample_axis_aligned_rotations
from cage_deform.Method.sample import sampleFibonacciRotations


def _run_single_icp(
    R_axis: np.ndarray,
    R_fib: np.ndarray,
    source_pts_coarse: np.ndarray,
    target_pcd_coarse: o3d.geometry.PointCloud,
    target_center: np.ndarray,
    threshold: float,
    with_scaling: bool=True,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """单次粗匹配 ICP，返回 (fitness, inlier_rmse, R_combined, transformation)。"""
    R_combined = R_fib @ R_axis
    rotated_source_coarse = (source_pts_coarse - target_center) @ R_combined.T + target_center
    source_pcd_coarse = toPcd(rotated_source_coarse)
    trans_init = np.eye(4)
    reg = o3d.pipelines.registration.registration_icp(
        source_pcd_coarse,
        target_pcd_coarse,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=with_scaling),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    return (reg.fitness, reg.inlier_rmse, R_combined.copy(), reg.transformation.copy())


class RigidMatcher(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def robustICP(
        source_points: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        test_rotation_num: int = 36,
        coarse_sample_num: int = 8192,
        threshold_ratio: float = 0.02,
        with_scaling: bool=True,
    ) -> np.ndarray:
        """鲁棒 ICP 配准，返回将 source 对齐到 target 的 4x4 变换矩阵。
        约定：点云 (N,4) 右乘 T 即得变换结果，无需转置：aligned = points @ T。"""
        source_pts = toNumpy(source_points).reshape(-1, 3)
        target_pts = toNumpy(target_points).reshape(-1, 3)

        target_pcd_full = toPcd(target_pts)

        # 剔除 target 离群点：只保留到球心距离前 95% 的点（球心用 bbox 中心）
        target_bbox_min, target_bbox_max = np.min(target_pts, axis=0), np.max(target_pts, axis=0)
        target_center = (target_bbox_min + target_bbox_max) / 2
        target_dists = np.linalg.norm(target_pts - target_center, axis=1)
        dist_95 = np.percentile(target_dists, 95)
        inlier_mask = target_dists <= dist_95
        target_pts = target_pts[inlier_mask]
        # 用内点 bbox 重新估计球心
        target_bbox_min, target_bbox_max = np.min(target_pts, axis=0), np.max(target_pts, axis=0)
        target_center = (target_bbox_min + target_bbox_max) / 2
        target_radius = np.percentile(np.linalg.norm(target_pts - target_center, axis=1), 95)

        # 将 source 球归一化并变换到 target 对应的球（球心用 bbox 中心）
        source_bbox_min, source_bbox_max = np.min(source_pts, axis=0), np.max(source_pts, axis=0)
        source_center = (source_bbox_min + source_bbox_max) / 2
        source_dists = np.linalg.norm(source_pts - source_center, axis=1)
        source_radius = np.percentile(source_dists, 95)
        source_radius = max(source_radius, 1e-9)  # 避免除零
        scale = target_radius / source_radius
        # 4x4: T_normalize: p -> scale*(p - source_center) + target_center
        T_normalize = np.eye(4)
        T_normalize[:3, :3] = scale * np.eye(3)
        T_normalize[:3, 3] = target_center - scale * source_center
        source_pts = (source_pts - source_center) / source_radius * target_radius + target_center

        # 粗匹配：最远点采样 + 大量旋转角度
        source_pcd = toPcd(source_pts)
        if len(source_pcd.points) > coarse_sample_num:
            source_pts_coarse = np.asarray(source_pcd.farthest_point_down_sample(coarse_sample_num).points)
        else:
            source_pts_coarse = np.asarray(source_pcd.points)

        if len(target_pcd_full.points) > coarse_sample_num:
            target_pcd_coarse = target_pcd_full.farthest_point_down_sample(coarse_sample_num)
        else:
            target_pcd_coarse = target_pcd_full

        axis_rotations = sample_axis_aligned_rotations()  # X/Y/Z 各 0,90,180,270 度，共 64 种
        fib_rotations = sampleFibonacciRotations(test_rotation_num)  # 每种轴位姿下的细采样旋转

        # 所有 (R_axis, R_fib) 组合，用于并行 ICP
        rotation_pairs = [(R_axis, R_fib) for R_axis in axis_rotations for R_fib in fib_rotations]
        total_cases = len(rotation_pairs)
        max_workers = min(32, (os.cpu_count() or 4) + 4)
        print(f"粗匹配: {coarse_sample_num} 点, 64 种轴对齐 × {test_rotation_num} 细旋转 = {total_cases} 次 ICP (并行 workers={max_workers})...")

        best_fitness = -1.0
        best_rmse = float("inf")
        best_rotation = None
        best_coarse_trans = None

        threshold = max(threshold_ratio * target_radius, 1e-9)

        def _task(args: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float, np.ndarray, np.ndarray]:
            R_axis, R_fib = args
            return _run_single_icp(
                R_axis, R_fib,
                source_pts_coarse, target_pcd_coarse, target_center, threshold, with_scaling,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_task, pair): pair for pair in rotation_pairs}
            for future in tqdm(as_completed(futures), total=total_cases, desc="粗匹配 ICP"):
                fitness, inlier_rmse, R_combined, trans = future.result()
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_rmse = inlier_rmse
                    best_rotation = R_combined
                    best_coarse_trans = trans

        if best_rotation is not None:
            print(f"  粗匹配最佳: Fitness={best_fitness:.6f}, RMSE={best_rmse:.6f}")

        if best_rotation is None:
            # 无有效配准时仅返回归一化变换（source 球对齐到 target 球），右乘约定
            return T_normalize.T.astype(np.float64)

        # 精匹配：用全点云，以粗匹配结果为初值做一次 ICP
        rotated_source_pts = (source_pts - target_center) @ best_rotation.T + target_center
        source_pcd_full = toPcd(rotated_source_pts)
        target_pcd_full = toPcd(target_pts)

        print("精匹配: 全点云 ICP...")
        reg_fine = o3d.pipelines.registration.registration_icp(
            source_pcd_full, target_pcd_full, threshold, best_coarse_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=with_scaling),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000),
        )

        # 组合为从原始 source 到最终对齐的 4x4 变换: T_fine @ T_rotation @ T_normalize
        # 精匹配输入为 rotated_source_pts = (p - c) @ best_rotation.T + c，即列向量下 p_rot = best_rotation @ (p - c) + c，
        # 故 T_rotation 应为“施加 best_rotation”的 4x4（列向量 p'=T@p），与 Open3D 一致。
        T_rotation = np.eye(4)
        T_rotation[:3, :3] = best_rotation
        T_rotation[:3, 3] = target_center - best_rotation @ target_center

        # 内部为列向量约定 p'=T@p；返回右乘约定，使 points @ T 即得变换结果
        T_final_col = reg_fine.transformation @ T_rotation @ T_normalize
        T_final = T_final_col.T
        print(f"  精匹配 Fitness={reg_fine.fitness:.6f}, RMSE={reg_fine.inlier_rmse:.6f}")

        return T_final.astype(np.float64)

    @staticmethod
    def robustICPFile(
        source_pcd_file_path: str,
        target_pcd_file_path: str,
        test_rotation_num: int = 36,
        coarse_sample_num: int = 8192,
        with_scaling: bool=True,
    ) -> np.ndarray:
        source_pcd = o3d.io.read_point_cloud(source_pcd_file_path)
        target_pcd = o3d.io.read_point_cloud(target_pcd_file_path)

        source_points = np.asarray(source_pcd.points)
        target_points = np.asarray(target_pcd.points)

        return RigidMatcher.robustICP(
            source_points,
            target_points,
            test_rotation_num,
            coarse_sample_num,
            with_scaling,
        )
