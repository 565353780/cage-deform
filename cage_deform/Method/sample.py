import numpy as np
from itertools import product


def _rotation_matrix_x(deg: float) -> np.ndarray:
    """绕 X 轴旋转 (度)."""
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_matrix_y(deg: float) -> np.ndarray:
    """绕 Y 轴旋转 (度)."""
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_matrix_z(deg: float) -> np.ndarray:
    """绕 Z 轴旋转 (度)."""
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def sample_axis_aligned_rotations() -> list[np.ndarray]:
    """生成 X/Y/Z 轴分别取 0, 90, 180, 270 度的所有旋转矩阵 (共 4^3=64 个)."""
    angles = (0, 90, 180, 270)
    rotations = []
    for rx, ry, rz in product(angles, angles, angles):
        R = _rotation_matrix_z(rz) @ _rotation_matrix_y(ry) @ _rotation_matrix_x(rx)
        rotations.append(R)
    return rotations

def sampleFibonacciPolars(num_polars: int) -> np.ndarray:
    """
    使用Fibonacci球面采样生成均匀分布的极角 (phi, theta)。

    Args:
        num_polars: 极角对的数量

    Returns:
        polars: shape (num_polars, 2) 的数组，每行为 (phi, theta)；
               phi 为极角/天顶角 [0, pi]，theta 为方位角。
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(num_polars, dtype=np.float64)
    if num_polars > 1:
        y = 1.0 - (i / (num_polars - 1)) * 2.0
    else:
        y = np.array([0.0])
    phi = np.arccos(np.clip(y, -1.0, 1.0))
    theta = golden_angle * i
    return np.stack([phi, theta], axis=1)

def sampleFibonacciRotations(num_rotations: int) -> np.ndarray:
    """
    基于 sampleFibonacciPolars 并行生成 num_rotations 个 3x3 旋转矩阵。
    R = Rz(phi) @ Ry(theta)，phi 为方位角 (xy 平面)，theta 为极角 (与 z 轴夹角)。

    Args:
        num_rotations: 旋转数量

    Returns:
        R: shape (num_rotations, 3, 3) 的旋转矩阵数组
    """
    polars = sampleFibonacciPolars(num_rotations)  # (n, 2), (phi, theta)
    phi = polars[:, 0]
    theta = polars[:, 1]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    n = num_rotations
    Rz = np.zeros((n, 3, 3), dtype=np.float64)
    Rz[:, 0, 0] = cphi
    Rz[:, 0, 1] = -sphi
    Rz[:, 1, 0] = sphi
    Rz[:, 1, 1] = cphi
    Rz[:, 2, 2] = 1.0
    Ry = np.zeros((n, 3, 3), dtype=np.float64)
    Ry[:, 0, 0] = cth
    Ry[:, 0, 2] = sth
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sth
    Ry[:, 2, 2] = cth
    return np.matmul(Rz, Ry)
