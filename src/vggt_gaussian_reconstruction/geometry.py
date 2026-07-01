from __future__ import annotations

import numpy as np
import torch


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = qvec.astype(np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(rotmat: np.ndarray) -> np.ndarray:
    m = rotmat.astype(np.float64)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def skew(v: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            torch.stack([zero, -vz, vy], dim=-1),
            torch.stack([vz, zero, -vx], dim=-1),
            torch.stack([-vy, vx, zero], dim=-1),
        ],
        dim=-2,
    )


def so3_exp_map(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-12)
    axis = rotvec / theta
    k = skew(axis)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(rotvec.shape[:-1] + (3, 3))
    theta_m = theta[..., None]
    return eye + torch.sin(theta_m) * k + (1.0 - torch.cos(theta_m)) * (k @ k)


def project(points: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor, intr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cam = (rot @ points[..., None]).squeeze(-1) + trans
    z = cam[..., 2].clamp_min(1e-8)
    uv_h = (intr @ cam[..., None]).squeeze(-1)
    uv = uv_h[..., :2] / z[..., None]
    return uv, cam[..., 2]
