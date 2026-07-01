from __future__ import annotations

import numpy as np

from vggt_gaussian_reconstruction.ba import optimize_reconstruction
from vggt_gaussian_reconstruction.colmap import Camera, Image, Point3D, Reconstruction
from vggt_gaussian_reconstruction.geometry import rotmat_to_qvec


def test_ba_reduces_synthetic_reprojection_error():
    camera = Camera(1, "PINHOLE", 640, 480, np.array([500.0, 500.0, 320.0, 240.0]))
    points = {
        i + 1: Point3D(
            i + 1,
            np.array([x, y, 4.0 + 0.2 * i], dtype=np.float64),
            np.array([255, 255, 255], dtype=np.uint8),
            0.0,
            [(1, i), (2, i)],
        )
        for i, (x, y) in enumerate([(-0.5, -0.3), (0.4, -0.2), (-0.2, 0.5), (0.6, 0.3)])
    }
    q_identity = rotmat_to_qvec(np.eye(3))
    q2 = rotmat_to_qvec(np.eye(3))
    true_t1 = np.array([0.0, 0.0, 0.0])
    true_t2 = np.array([-0.4, 0.0, 0.0])
    noisy_t2 = np.array([-0.32, 0.03, 0.0])

    xys1 = []
    xys2 = []
    for p in points.values():
        xys1.append(project_np(p.xyz, true_t1))
        xys2.append(project_np(p.xyz, true_t2))
    images = {
        1: Image(1, q_identity.copy(), true_t1.copy(), 1, "im1.png", np.array(xys1), np.array([1, 2, 3, 4])),
        2: Image(2, q2.copy(), noisy_t2.copy(), 1, "im2.png", np.array(xys2), np.array([1, 2, 3, 4])),
    }
    model = Reconstruction({1: camera}, images, points)
    _, stats = optimize_reconstruction(model, iters=80, lr_pose=5e-3, lr_points=1e-3)
    assert stats.final_rmse < stats.initial_rmse


def project_np(xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    cam = xyz + t
    return np.array([500.0 * cam[0] / cam[2] + 320.0, 500.0 * cam[1] / cam[2] + 240.0])
