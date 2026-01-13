"""
三种空间变形器的演示代码：

1. BSplineDeformer - 基于三次 B 样条的 FFD，使用 4×4×4 控制点进行插值
   - 优点：C² 连续，变形平滑无折痕
   - 适用：需要精确控制变形场的场景

2. RBFDeformer - 径向基函数变形
   - 优点：实现简单，可达 C^∞ 光滑，不需要构建网格
   - 适用：控制点数量较少（< 1000）的场景

3. MLSDeformer - 移动最小二乘变形
   - 优点：刚性模式可以很好地保持曲率，不需要求解全局方程
   - 适用：需要保持局部形状的变形场景
"""

import torch
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not available, visualization will be skipped")

from cage_deform.Method.data import toTensor
from cage_deform.Module.bspline_deformer import BSplineDeformer
from cage_deform.Module.rbf_deformer import RBFDeformer
from cage_deform.Module.mls_deformer import MLSDeformer


def generate_test_data(dtype=torch.float32, device='cpu'):
    """生成测试数据：一个简单的点云和变形约束"""
    # 生成一个球面点云
    n_points = 1000
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    r = 1.0
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    
    # 选择一些控制点（球面上的关键点）
    n_controls = 20
    control_indices = np.random.choice(n_points, n_controls, replace=False)
    
    source_controls = points[control_indices]
    
    # 目标：对控制点应用一个弯曲变形
    target_controls = source_controls.copy()
    # 将 z > 0 的点向外推，z < 0 的点向内收
    target_controls[:, 0] += 0.2 * source_controls[:, 2]
    target_controls[:, 1] += 0.1 * source_controls[:, 2]
    
    return {
        'points': toTensor(points, dtype, device),
        'source_controls': toTensor(source_controls, dtype, device),
        'target_controls': toTensor(target_controls, dtype, device),
    }


def demo_bspline():
    """演示 B 样条 FFD 变形"""
    print("\n" + "="*60)
    print("B-Spline FFD Demo")
    print("="*60)
    
    dtype = torch.float32
    device = 'cpu'
    
    data = generate_test_data(dtype, device)
    
    # 创建 B 样条变形器
    deformer = BSplineDeformer(dtype=dtype, device=device)
    
    # 加载点云并创建控制网格
    grid_info = deformer.loadPoints(
        data['points'],
        voxel_size=0.25,
        padding=0.1
    )
    print(f"Control grid info: {grid_info}")
    
    # 优化变形场
    deformed_controls = deformer.deformPoints(
        data['source_controls'],
        data['target_controls'],
        lr=1e-2,
        lambda_smooth=1e3,
        lambda_magnitude=1.0,
        steps=200
    )
    
    # 查询整个点云的变形结果
    deformed_points = deformer.queryPoints(data['points'])
    
    print(f"Input points shape: {data['points'].shape}")
    print(f"Deformed points shape: {deformed_points.shape}")
    
    return deformed_points


def demo_rbf():
    """演示 RBF 变形"""
    print("\n" + "="*60)
    print("RBF Deformation Demo")
    print("="*60)
    
    dtype = torch.float32
    device = 'cpu'
    
    data = generate_test_data(dtype, device)
    
    # 创建 RBF 变形器（使用 triharmonic 核函数）
    deformer = RBFDeformer(
        kernel='triharmonic',  # 也可以用 'gaussian', 'thin_plate' 等
        regularization=1e-4,
        dtype=dtype,
        device=device
    )
    
    # 拟合变形场
    deformer.fit(
        data['source_controls'],
        data['target_controls'],
        use_affine_init=True
    )
    
    # 变换整个点云
    deformed_points = deformer.transform(data['points'])
    
    print(f"Input points shape: {data['points'].shape}")
    print(f"Deformed points shape: {deformed_points.shape}")
    
    # 也可以用一行代码完成：
    # deformed_points = deformer.deformPoints(
    #     data['source_controls'],
    #     data['target_controls'],
    #     query_points=data['points']
    # )
    
    return deformed_points


def demo_mls():
    """演示 MLS 变形"""
    print("\n" + "="*60)
    print("MLS Deformation Demo")
    print("="*60)
    
    dtype = torch.float32
    device = 'cpu'
    
    data = generate_test_data(dtype, device)
    
    # 创建 MLS 变形器（使用刚性模式保持局部形状）
    deformer = MLSDeformer(
        mode='rigid',  # 也可以用 'affine', 'similarity'
        alpha=2.0,     # 权重衰减指数，越大变形越局部化
        dtype=dtype,
        device=device
    )
    
    # 设置控制点
    deformer.fit(
        data['source_controls'],
        data['target_controls'],
        use_affine_init=True
    )
    
    # 变换整个点云
    deformed_points = deformer.transform(data['points'])
    
    print(f"Input points shape: {data['points'].shape}")
    print(f"Deformed points shape: {deformed_points.shape}")
    
    # 获取局部变换参数
    rotations, translations = deformer.get_local_transform(data['points'][:10])
    print(f"Local rotations shape: {rotations.shape}")
    print(f"Local translations shape: {translations.shape}")
    
    return deformed_points


def visualize_comparison(original, bspline_result, rbf_result, mls_result):
    """可视化对比三种方法的结果"""
    if not HAS_OPEN3D:
        print("Skipping visualization (open3d not available)")
        return
    
    def to_o3d_pcd(points, color):
        pcd = o3d.geometry.PointCloud()
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        return pcd
    
    # 创建点云，用不同颜色区分
    pcds = [
        to_o3d_pcd(original, [0.5, 0.5, 0.5]),      # 灰色：原始
        to_o3d_pcd(bspline_result, [1, 0, 0]),      # 红色：B样条
        to_o3d_pcd(rbf_result + np.array([2.5, 0, 0]), [0, 1, 0]),  # 绿色：RBF
        to_o3d_pcd(mls_result + np.array([5, 0, 0]), [0, 0, 1]),    # 蓝色：MLS
    ]
    
    # 添加坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    print("\nVisualization colors:")
    print("  Gray: Original")
    print("  Red: B-Spline FFD")
    print("  Green: RBF (offset +2.5 in X)")
    print("  Blue: MLS (offset +5 in X)")
    
    o3d.visualization.draw_geometries(pcds + [coord_frame])


def demo():
    """运行所有演示"""
    print("="*60)
    print("Smooth Deformation Methods Comparison")
    print("="*60)
    
    dtype = torch.float32
    device = 'cpu'
    
    # 生成测试数据
    data = generate_test_data(dtype, device)
    original_points = data['points']
    
    # 运行三种方法
    bspline_result = demo_bspline()
    rbf_result = demo_rbf()
    mls_result = demo_mls()
    
    # 可视化对比
    visualize_comparison(original_points, bspline_result, rbf_result, mls_result)
    
    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)
    
    return True


if __name__ == '__main__':
    demo()
