from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .colmap import Image, Reconstruction, camera_matrix
from .geometry import project, qvec_to_rotmat, rotmat_to_qvec, so3_exp_map


@dataclass
class BAStats:
    initial_rmse: float
    final_rmse: float
    observations: int
    optimized_points: int


def optimize_reconstruction(
    model: Reconstruction,
    iters: int = 1000,
    lr_pose: float = 1e-3,
    lr_points: float = 1e-2,
    huber_delta: float = 4.0,
    min_track_len: int = 2,
    device: str = "cpu",
) -> tuple[Reconstruction, BAStats]:
    observations = _build_observations(model, min_track_len=min_track_len)
    if len(observations["image_indices"]) == 0:
        raise ValueError("No valid 2D-3D observations found for BA")

    image_ids = sorted(model.images)
    point_ids = observations["point_ids"]
    image_index = {image_id: i for i, image_id in enumerate(image_ids)}
    point_index = {point_id: i for i, point_id in enumerate(point_ids)}

    dtype = torch.float64
    dev = torch.device(device)
    base_rot = torch.tensor(
        np.stack([qvec_to_rotmat(model.images[i].qvec) for i in image_ids]),
        dtype=dtype,
        device=dev,
    )
    base_trans = torch.tensor(
        np.stack([model.images[i].tvec for i in image_ids]),
        dtype=dtype,
        device=dev,
    )
    points0 = torch.tensor(
        np.stack([model.points3d[i].xyz for i in point_ids]),
        dtype=dtype,
        device=dev,
    )
    intr = torch.tensor(
        np.stack([camera_matrix(model.cameras[model.images[i].camera_id]) for i in image_ids]),
        dtype=dtype,
        device=dev,
    )
    obs_image = torch.tensor([image_index[i] for i in observations["image_indices"]], dtype=torch.long, device=dev)
    obs_point = torch.tensor([point_index[i] for i in observations["point_ids_per_obs"]], dtype=torch.long, device=dev)
    obs_xy = torch.tensor(observations["xy"], dtype=dtype, device=dev)

    pose_delta = torch.zeros((len(image_ids), 6), dtype=dtype, device=dev, requires_grad=True)
    points = points0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": [pose_delta], "lr": lr_pose},
            {"params": [points], "lr": lr_points},
        ]
    )

    with torch.no_grad():
        initial_rmse = _rmse(points, base_rot, base_trans, intr, obs_image, obs_point, obs_xy).item()

    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        rot_delta = so3_exp_map(pose_delta[:, :3])
        rot = rot_delta @ base_rot
        trans = base_trans + pose_delta[:, 3:6]
        uv, depth = project(points[obs_point], rot[obs_image], trans[obs_image], intr[obs_image])
        valid = depth > 1e-6
        residual = uv - obs_xy
        if not torch.any(valid):
            raise RuntimeError("All BA observations moved behind cameras")
        loss = torch.nn.functional.huber_loss(residual[valid], torch.zeros_like(residual[valid]), delta=huber_delta)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        rot_delta = so3_exp_map(pose_delta[:, :3])
        final_rot = (rot_delta @ base_rot).detach().cpu().numpy()
        final_trans = (base_trans + pose_delta[:, 3:6]).detach().cpu().numpy()
        final_points = points.detach().cpu().numpy()
        final_rmse = _rmse(points, rot_delta @ base_rot, base_trans + pose_delta[:, 3:6], intr, obs_image, obs_point, obs_xy).item()

    refined = _copy_model(model)
    for i, image_id in enumerate(image_ids):
        refined.images[image_id].qvec = rotmat_to_qvec(final_rot[i])
        refined.images[image_id].tvec = final_trans[i]
    for i, point_id in enumerate(point_ids):
        refined.points3d[point_id].xyz = final_points[i]
        refined.points3d[point_id].error = final_rmse

    return refined, BAStats(initial_rmse, final_rmse, len(obs_image), len(point_ids))


def _rmse(
    points: torch.Tensor,
    rot: torch.Tensor,
    trans: torch.Tensor,
    intr: torch.Tensor,
    obs_image: torch.Tensor,
    obs_point: torch.Tensor,
    obs_xy: torch.Tensor,
) -> torch.Tensor:
    uv, depth = project(points[obs_point], rot[obs_image], trans[obs_image], intr[obs_image])
    residual = uv - obs_xy
    valid = depth > 1e-6
    return torch.sqrt(torch.mean(torch.sum(residual[valid] ** 2, dim=-1)))


def _build_observations(model: Reconstruction, min_track_len: int) -> dict:
    image_indices: list[int] = []
    point_ids_per_obs: list[int] = []
    xy: list[np.ndarray] = []
    usable_point_ids = {
        point_id
        for point_id, point in model.points3d.items()
        if len(point.track) >= min_track_len and np.all(np.isfinite(point.xyz))
    }
    for image_id, image in model.images.items():
        for point2d_idx, point_id in enumerate(image.point3d_ids):
            pid = int(point_id)
            if pid == -1 or pid not in usable_point_ids:
                continue
            if point2d_idx >= len(image.xys):
                continue
            xy_val = image.xys[point2d_idx]
            if not np.all(np.isfinite(xy_val)):
                continue
            image_indices.append(image_id)
            point_ids_per_obs.append(pid)
            xy.append(xy_val.astype(np.float64))
    return {
        "image_indices": image_indices,
        "point_ids": sorted(set(point_ids_per_obs)),
        "point_ids_per_obs": point_ids_per_obs,
        "xy": np.stack(xy) if xy else np.zeros((0, 2), dtype=np.float64),
    }


def _copy_model(model: Reconstruction) -> Reconstruction:
    cameras = {k: v for k, v in model.cameras.items()}
    images = {
        k: Image(
            id=v.id,
            qvec=v.qvec.copy(),
            tvec=v.tvec.copy(),
            camera_id=v.camera_id,
            name=v.name,
            xys=v.xys.copy(),
            point3d_ids=v.point3d_ids.copy(),
        )
        for k, v in model.images.items()
    }
    points = {
        k: type(v)(id=v.id, xyz=v.xyz.copy(), rgb=v.rgb.copy(), error=v.error, track=list(v.track))
        for k, v in model.points3d.items()
    }
    return Reconstruction(cameras, images, points)
