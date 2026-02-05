import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union

from cage_deform.Method.pcd import toPcd
from cage_deform.Method.data import toNumpy
from cage_deform.Method.io import loadMeshFile
from cage_deform.Method.path import createFileFolder
from cage_deform.Method.sample import sample_axis_aligned_rotations
from cage_deform.Method.sample import sampleFibonacciRotations


class RigidMatcher(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def robustICP(
        source_points: Union[torch.Tensor, np.ndarray, list],
        target_points: Union[torch.Tensor, np.ndarray, list],
        test_rotation_num: int = 36,
        coarse_sample_num: int = 8192,
    ) -> np.ndarray:
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
        threshold = 0.02  # 移动范围阈值
        trans_init = np.eye(4)

        best_fitness = -1.0
        best_rotation = None
        best_coarse_trans = None

        total_cases = len(axis_rotations) * len(fib_rotations)
        print(f"粗匹配: {coarse_sample_num} 点, 64 种轴对齐 × {test_rotation_num} 细旋转 = {total_cases} 次 ICP...")
        for R_axis in tqdm(axis_rotations, desc="64 轴对齐"):
            for R_fib in fib_rotations:
                # 组合旋转：先轴对齐 R_axis，再细旋转 R_fib → 总旋转 R = R_fib @ R_axis
                R_combined = R_fib @ R_axis
                rotated_source_coarse = (source_pts_coarse - target_center) @ R_combined.T + target_center
                source_pcd_coarse = toPcd(rotated_source_coarse)

                reg = o3d.pipelines.registration.registration_icp(
                    source_pcd_coarse, target_pcd_coarse, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
                )

                if reg.fitness > best_fitness:
                    best_fitness = reg.fitness
                    best_rotation = R_combined.copy()
                    best_coarse_trans = reg.transformation.copy()
                    print(f"  更新最佳: Fitness={reg.fitness:.6f}, RMSE={reg.inlier_rmse:.6f}")

        if best_rotation is None:
            best_icp_pts = source_pts  # 无有效配准时退回当前 source_pts
        else:
            # 精匹配：用全点云，以粗匹配结果为初值做一次 ICP
            rotated_source_pts = (source_pts - target_center) @ best_rotation.T + target_center
            source_pcd_full = toPcd(rotated_source_pts)
            target_pcd_full = toPcd(target_pts)

            print("精匹配: 全点云 ICP...")
            reg_fine = o3d.pipelines.registration.registration_icp(
                source_pcd_full, target_pcd_full, threshold, best_coarse_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000),
            )
            source_pcd_full.transform(reg_fine.transformation)
            best_icp_pts = np.asarray(source_pcd_full.points)
            print(f"  精匹配 Fitness={reg_fine.fitness:.6f}, RMSE={reg_fine.inlier_rmse:.6f}")

        return best_icp_pts

    @staticmethod
    def robustICPFile(
        source_mesh_file_path: str,
        target_pcd_file_path: str,
        save_mesh_file_path: str,
        test_rotation_num: int = 36,
        coarse_sample_num: int = 8192,
    ) -> bool:
        if os.path.exists(save_mesh_file_path):
            return True

        source_mesh = loadMeshFile(source_mesh_file_path)
        if source_mesh is None:
            print('[ERROR][RigidMatcher::robustICPFile]')
            print('\t loadMeshFile failed!')
            return False

        target_pcd = o3d.io.read_point_cloud(target_pcd_file_path)

        source_points = source_mesh.vertices
        target_points = np.asarray(target_pcd.points)

        best_icp_pts = RigidMatcher.robustICP(
            source_points,
            target_points,
            test_rotation_num,
            coarse_sample_num,
        )

        source_mesh.vertices = best_icp_pts

        createFileFolder(save_mesh_file_path)
        source_mesh.export(save_mesh_file_path)
        return True
