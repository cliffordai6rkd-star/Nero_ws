from __future__ import annotations

import math

import numpy as np


def pose6_to_matrix(pose: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    mat = np.eye(4, dtype=np.float64)
    if values.size < 3:
        return mat
    mat[:3, 3] = values[:3]
    if values.size >= 6:
        mat[:3, :3] = euler_xyz_to_matrix(values[3], values[4], values[5])
    return mat


def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_mat = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_mat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_mat @ ry_mat @ rx_mat


def matrix_from_joint_stub(q: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    if q.size:
        mat[0, 3] = 0.25 + 0.02 * float(np.sum(np.cos(q)))
        mat[1, 3] = 0.02 * float(np.sum(np.sin(q)))
        mat[2, 3] = 0.15 + 0.01 * float(q[0])
    return mat
