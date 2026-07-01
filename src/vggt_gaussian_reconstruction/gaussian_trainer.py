from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from tqdm import tqdm

from .colmap import camera_matrix, read_model
from .geometry import qvec_to_rotmat, rotmat_to_qvec


@dataclass
class TrainConfig:
    scene: Path
    sparse_dir: Path
    image_dir: Path
    output_dir: Path
    steps: int = 30000
    device: str = "cuda"
    lr: float = 1.0
    image_scale: float = 1.0
    max_points: int = 200_000
    save_every: int = 1000
    render_every: int = 1000
    normalize_world_space: bool = True
    init_opacity: float = 0.1
    packed: bool = False
    random_background: bool = False
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_scale: float = 1.0
    ssim_lambda: float = 0.2
    antialiased: bool = False
    near_plane: float = 0.01
    far_plane: float = 1e10
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20


def train_gaussians(config: TrainConfig) -> None:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model = read_model(config.sparse_dir)
    if config.normalize_world_space:
        model = _normalize_model(model)
    scene_scale = _camera_scene_scale(model) * 1.1

    cameras, images, targets = _load_training_views(model, config.image_dir, config.image_scale, device)
    means_np, colors_np = _initial_points(model, config.max_points)
    if len(means_np) == 0:
        raise ValueError(f"No usable 3D points found in {config.sparse_dir}")

    from gsplat.strategy import DefaultStrategy

    means0 = torch.as_tensor(means_np, dtype=torch.float32, device=device)
    rgbs0 = torch.as_tensor(colors_np, dtype=torch.float32, device=device)
    colors0 = torch.zeros((means0.shape[0], (config.sh_degree + 1) ** 2, 3), dtype=torch.float32, device=device)
    colors0[:, 0, :] = _rgb_to_sh(rgbs0)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means0),
            "scales": torch.nn.Parameter(_initial_scales(means_np, config.init_scale, device)),
            "quats": torch.nn.Parameter(_initial_quats(means_np.shape[0], device)),
            "opacities": torch.nn.Parameter(
                torch.full((means_np.shape[0],), _logit(config.init_opacity), dtype=torch.float32, device=device)
            ),
            "sh0": torch.nn.Parameter(colors0[:, :1, :].contiguous()),
            "shN": torch.nn.Parameter(colors0[:, 1:, :].contiguous()),
        }
    )
    optimizers = {
        "means": _make_adam(params["means"], config.lr * config.means_lr * scene_scale, "means", device),
        "scales": _make_adam(params["scales"], config.lr * config.scales_lr, "scales", device),
        "quats": _make_adam(params["quats"], config.lr * config.quats_lr, "quats", device),
        "opacities": _make_adam(params["opacities"], config.lr * config.opacities_lr, "opacities", device),
        "sh0": _make_adam(params["sh0"], config.lr * config.sh0_lr, "sh0", device),
        "shN": _make_adam(params["shN"], config.lr * config.shN_lr, "shN", device),
    }
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max(config.steps, 1))
        )
    ]
    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = config.output_dir / "renders"
    render_dir.mkdir(exist_ok=True)

    view_order: list[int] = []
    pbar = tqdm(range(config.steps), desc="Training Gaussians")
    for step in pbar:
        if not view_order:
            view_order = torch.randperm(len(images)).tolist()
        view_idx = view_order.pop()
        sh_degree_to_use = min(step // max(config.sh_degree_interval, 1), config.sh_degree)
        render, _alpha, info = _render(params, cameras[view_idx], config, sh_degree_to_use)
        if config.random_background:
            background = torch.rand((1, 1, 3), dtype=render.dtype, device=render.device)
            render = render + background * (1.0 - _alpha)
        target = targets[view_idx]
        l1loss = F.l1_loss(render, target)
        ssimloss = _ssim_loss(render[None].permute(0, 3, 1, 2), target[None].permute(0, 3, 1, 2))
        loss = torch.lerp(l1loss, ssimloss, config.ssim_lambda)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            _prepare_optimizer_grads(optimizer)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=config.packed)

        step_num = step + 1
        if step == 0 or step_num % 50 == 0:
            pbar.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                l1=f"{float(l1loss.detach()):.4f}",
                ssim=f"{float(ssimloss.detach()):.4f}",
                sh=sh_degree_to_use,
                image=images[view_idx].name,
                gaussians=len(params["means"]),
            )
        if step_num % config.save_every == 0 or step_num == config.steps:
            _save_checkpoint(config.output_dir / "checkpoint.pt", step_num, params)
        if step_num % config.render_every == 0 or step_num == config.steps:
            _write_render(render_dir / images[view_idx].name, render.detach())


def _load_training_views(model, image_dir: Path, scale: float, device: torch.device):
    cameras = []
    images = []
    targets = []
    for image in sorted(model.images.values(), key=lambda x: x.name):
        camera = model.cameras[image.camera_id]
        path = image_dir / image.name
        if not path.exists():
            continue
        target = _load_image(path, scale, device)
        height, width = target.shape[:2]
        k = torch.as_tensor(camera_matrix(camera), dtype=torch.float32, device=device)
        k[0, :] *= width / camera.width
        k[1, :] *= height / camera.height
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = qvec_to_rotmat(image.qvec).astype(np.float32)
        viewmat[:3, 3] = image.tvec.astype(np.float32)
        camtoworld = np.linalg.inv(viewmat)
        cameras.append(
            {
                "K": k,
                "camtoworld": torch.as_tensor(camtoworld, dtype=torch.float32, device=device),
                "width": width,
                "height": height,
            }
        )
        images.append(image)
        targets.append(target)
    if not targets:
        raise ValueError(f"No training images found under {image_dir}")
    return cameras, images, targets


def _initial_points(model, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    points = [p for p in model.points3d.values() if np.all(np.isfinite(p.xyz)) and len(p.track) >= 2]
    points = _filter_point_outliers(points)
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = [points[i] for i in keep]
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    means = np.stack([p.xyz for p in points], axis=0).astype(np.float32)
    colors = np.stack([p.rgb.astype(np.float32) / 255.0 for p in points], axis=0).astype(np.float32)
    return means, colors


def _filter_point_outliers(points: list) -> list:
    if len(points) < 100:
        return points

    xyz = np.stack([p.xyz for p in points], axis=0).astype(np.float64)
    center = np.median(xyz, axis=0, keepdims=True)
    radius = np.linalg.norm(xyz - center, axis=1)
    if not np.all(np.isfinite(radius)):
        finite = np.isfinite(radius)
        points = [p for p, keep in zip(points, finite) if keep]
        radius = radius[finite]
    if len(radius) < 100:
        return points

    q75 = float(np.quantile(radius, 0.75))
    q25 = float(np.quantile(radius, 0.25))
    iqr = max(q75 - q25, 1e-6)
    percentile_cutoff = float(np.quantile(radius, 0.999))
    iqr_cutoff = q75 + 10.0 * iqr
    cutoff = max(min(percentile_cutoff, iqr_cutoff), 1e-4)
    keep_mask = radius <= cutoff
    if keep_mask.all():
        return points
    kept = [p for p, keep in zip(points, keep_mask) if keep]
    print(f"Filtered {len(points) - len(kept)} outlier 3D points before Gaussian training; kept {len(kept)}.")
    return kept


def _render(params, camera, config: TrainConfig, sh_degree: int | None = None):
    from gsplat.rendering import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    colors, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv_ex(camera["camtoworld"][None]).inverse,
        Ks=camera["K"][None],
        width=camera["width"],
        height=camera["height"],
        near_plane=config.near_plane,
        far_plane=config.far_plane,
        sh_degree=sh_degree,
        packed=config.packed,
        absgrad=getattr(config, "absgrad", False),
        rasterize_mode="antialiased" if config.antialiased else "classic",
    )
    return colors[0, ..., :3].clamp(0.0, 1.0), alpha[0], info


def _save_checkpoint(path: Path, step: int, params) -> None:
    torch.save(
        {
            "step": step,
            "splats": params.state_dict(),
            "means": params["means"].detach().cpu(),
            "quats": params["quats"].detach().cpu(),
            "scales": torch.exp(params["scales"].detach()).cpu(),
            "opacities": torch.sigmoid(params["opacities"].detach()).cpu(),
            "sh0": params["sh0"].detach().cpu(),
            "shN": params["shN"].detach().cpu(),
            "colors": _sh_to_rgb(params["sh0"].detach()[:, 0, :]).clamp(0.0, 1.0).cpu(),
        },
        path,
    )


def _load_image(path: Path, scale: float, device: torch.device) -> torch.Tensor:
    with PILImage.open(path) as im:
        im = im.convert("RGB")
        if scale != 1.0:
            width = max(1, int(round(im.width * scale)))
            height = max(1, int(round(im.height * scale)))
            im = im.resize((width, height), PILImage.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _write_render(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    PILImage.fromarray(out).save(path.with_suffix(".png"))


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _make_adam(param: torch.nn.Parameter, lr: float, name: str, device: torch.device) -> torch.optim.Optimizer:
    kwargs = {"eps": 1e-15, "betas": (0.9, 0.999)}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.Adam([{"params": param, "lr": lr, "name": name}], **kwargs)


def _prepare_optimizer_grads(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for param in group["params"]:
            grad = param.grad
            if grad is None or grad.is_sparse:
                continue
            if grad.dtype != param.dtype or grad.device != param.device:
                grad = grad.to(device=param.device, dtype=param.dtype)
            if grad.layout != param.layout or not grad.is_contiguous():
                grad = grad.contiguous()
            param.grad = grad


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def _sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return sh * c0 + 0.5


def _initial_scale(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.01
    try:
        from scipy.spatial import cKDTree

        k = min(4, len(points))
        _, idx = cKDTree(points).query(points, k=k)
        neighbors = points[idx[:, 1:]]
        dist = np.linalg.norm(neighbors - points[:, None, :], axis=-1).mean(axis=1)
        dist = dist[np.isfinite(dist) & (dist > 0)]
        if len(dist) == 0:
            return 0.01
        return float(max(np.median(dist), 1e-4))
    except Exception:
        radius = np.linalg.norm(points - np.median(points, axis=0, keepdims=True), axis=1)
        radius = radius[np.isfinite(radius) & (radius > 0)]
        if len(radius) == 0:
            return 0.01
        return float(max(np.percentile(radius, 25) * 0.1, 1e-4))


def _initial_scales(points: np.ndarray, init_scale: float, device: torch.device) -> torch.Tensor:
    if len(points) == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)
    if len(points) < 4:
        scale = max(_initial_scale(points) * init_scale, 1e-4)
        return torch.full((len(points), 3), math.log(scale), dtype=torch.float32, device=device)
    try:
        from scipy.spatial import cKDTree

        dists, _ = cKDTree(points).query(points, k=4)
        dist2_avg = np.mean(dists[:, 1:] ** 2, axis=-1)
        dist_avg = np.sqrt(np.maximum(dist2_avg, 1e-8)) * init_scale
    except Exception:
        fallback = max(_initial_scale(points) * init_scale, 1e-4)
        dist_avg = np.full((len(points),), fallback, dtype=np.float32)
    dist_avg = np.maximum(dist_avg, 1e-4).astype(np.float32)
    scales = np.log(dist_avg)[:, None].repeat(3, axis=1)
    return torch.as_tensor(scales, dtype=torch.float32, device=device)


def _initial_quats(count: int, device: torch.device) -> torch.Tensor:
    return torch.rand((count, 4), dtype=torch.float32, device=device)


def _ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x * x, kernel_size=11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=11, stride=1, padding=5) - mu_x * mu_y
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return torch.clamp((1.0 - ssim) / 2.0, 0.0, 1.0).mean()


def _normalize_model(model):
    images = list(model.images.values())
    points = list(model.points3d.values())
    if not images or not points:
        return model

    image_ids = [image.id for image in images]
    camtoworlds = []
    for image in images:
        viewmat = np.eye(4, dtype=np.float64)
        viewmat[:3, :3] = qvec_to_rotmat(image.qvec)
        viewmat[:3, 3] = image.tvec.astype(np.float64)
        camtoworlds.append(np.linalg.inv(viewmat))
    camtoworlds_np = np.stack(camtoworlds, axis=0)
    points_np = np.stack([point.xyz for point in points], axis=0).astype(np.float64)

    transform1 = _similarity_from_cameras(camtoworlds_np)
    camtoworlds_np = _transform_cameras(transform1, camtoworlds_np)
    points_np = _transform_points(transform1, points_np)

    if len(points_np) >= 3:
        transform2 = _align_principal_axes(points_np)
        camtoworlds_np = _transform_cameras(transform2, camtoworlds_np)
        points_np = _transform_points(transform2, points_np)

    if np.median(points_np[:, 2]) > np.mean(points_np[:, 2]):
        transform3 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        camtoworlds_np = _transform_cameras(transform3, camtoworlds_np)
        points_np = _transform_points(transform3, points_np)

    normalized = type(model)(
        cameras={k: v for k, v in model.cameras.items()},
        images={},
        points3d={},
    )
    camtoworld_by_id = dict(zip(image_ids, camtoworlds_np))
    for image_id, image in model.images.items():
        camtoworld = camtoworld_by_id[image_id]
        viewmat = np.linalg.inv(camtoworld)
        normalized.images[image_id] = type(image)(
            id=image.id,
            qvec=rotmat_to_qvec(viewmat[:3, :3]),
            tvec=viewmat[:3, 3],
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys.copy(),
            point3d_ids=image.point3d_ids.copy(),
        )
    for point, xyz in zip(points, points_np):
        normalized.points3d[point.id] = type(point)(
            id=point.id,
            xyz=xyz,
            rgb=point.rgb.copy(),
            error=point.error,
            track=list(point.track),
        )
    return normalized


def _camera_scene_scale(model) -> float:
    images = list(model.images.values())
    if len(images) < 2:
        return 1.0
    camera_locations = []
    for image in images:
        rot = qvec_to_rotmat(image.qvec)
        camera_locations.append(-(rot.T @ image.tvec.astype(np.float64)))
    camera_locations_np = np.stack(camera_locations, axis=0)
    scene_center = np.mean(camera_locations_np, axis=0)
    dists = np.linalg.norm(camera_locations_np - scene_center[None, :], axis=1)
    scene_scale = float(np.max(dists)) if len(dists) else 1.0
    if not np.isfinite(scene_scale) or scene_scale <= 1e-8:
        return 1.0
    return scene_scale


def _similarity_from_cameras(camtoworlds: np.ndarray) -> np.ndarray:
    translations = camtoworlds[:, :3, 3]
    rotations = camtoworlds[:, :3, :3]

    ups = np.sum(rotations * np.array([0.0, -1.0, 0.0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up_norm = np.linalg.norm(world_up)
    if not np.isfinite(world_up_norm) or world_up_norm <= 1e-8:
        world_up = np.array([0.0, -1.0, 0.0])
    else:
        world_up /= world_up_norm

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = float((up_camspace * world_up).sum())
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        rotation_align = np.eye(3) + skew + (skew @ skew) * (1.0 / max(1.0 + c, 1e-8))
    else:
        rotation_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    rotations = rotation_align @ rotations
    forwards = np.sum(rotations * np.array([0.0, 0.0, 1.0]), axis=-1)
    translations = (rotation_align @ translations[..., None])[..., 0]
    nearest = translations + (forwards * -translations).sum(-1)[:, None] * forwards
    translate = -np.median(nearest, axis=0)

    denom = np.median(np.linalg.norm(translations + translate, axis=-1))
    scale = 1.0 / denom if np.isfinite(denom) and denom > 1e-8 else 1.0
    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = rotation_align
    transform[:3, :] *= scale
    return transform


def _align_principal_axes(points: np.ndarray) -> np.ndarray:
    centroid = np.median(points, axis=0)
    translated = points - centroid
    covariance = np.cov(translated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return transform


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    transformed = np.einsum("nij,ki->nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(transformed[:, 0, :3], axis=1)
    scaling = np.where(scaling > 1e-8, scaling, 1.0)
    transformed[:, :3, :3] = transformed[:, :3, :3] / scaling[:, None, None]
    return transformed
