from __future__ import annotations

import json
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
    opacities_lr: float = 2e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    test_every: int = 8
    max_gaussians: int = 200_000
    refine_stop_iter: int = 15_000
    grow_grad2d: float = 0.0002
    prune_opa: float = 0.01
    min_track_len: int = 4
    opacity_reg: float = 1e-3
    scale_reg: float = 1e-3
    prune_every: int = 1000
    prune_large_scale: float = 0.25
    visibility_prune_every: int = 2000
    visibility_prune_start: int = 8000
    visibility_min_views: int = 2
    prune_scene_radius: float = 2.5
    depth_loss: bool = True
    depth_lambda: float = 1e-2
    depth_sample_count: int = 2048
    depth_loss_clamp: float = 2.0
    mask_dir: Path | None = None
    mask_loss: bool = True
    mask_alpha_lambda: float = 0.05
    mask_threshold: float = 0.5
    opacity_reset_every: int = 1000
    opacity_reset_until: int = 0
    final_prune_opa: float = 0.05
    final_prune_large_scale: float = 0.08
    final_prune_scene_radius: float = 2.8
    final_visibility_min_views: int = 3
    distributed: bool = False


def train_gaussians(config: TrainConfig) -> None:
    if config.distributed:
        from gsplat.distributed import cli

        cli(_train_gaussians_worker, config, verbose=True)
        return
    _train_gaussians_worker(0, 0, 1, config)


def _train_gaussians_worker(local_rank: int, world_rank: int, world_size: int, config: TrainConfig) -> None:
    torch.manual_seed(42 + world_rank)
    np.random.seed(42 + world_rank)
    is_distributed = world_size > 1
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model = read_model(config.sparse_dir)
    if config.normalize_world_space:
        model = _normalize_model(model)
    scene_scale = _camera_scene_scale(model) * 1.1

    initial_points = _select_initial_points(model, config.max_points, config.min_track_len)
    allowed_depth_point_ids = {int(point.id) for point in initial_points}
    cameras, images, targets = _load_training_views(
        model,
        config.image_dir,
        config.image_scale,
        device,
        allowed_depth_point_ids=allowed_depth_point_ids,
        mask_dir=config.mask_dir,
        mask_threshold=config.mask_threshold,
    )
    train_indices, val_indices = _split_view_indices(len(images), config.test_every)
    means_np, colors_np = _points_to_arrays(initial_points)
    if len(means_np) == 0:
        raise ValueError(f"No usable 3D points found in {config.sparse_dir}")
    local_point_indices = np.arange(len(means_np))[world_rank::world_size]
    scales0_full = _initial_scales(means_np, config.init_scale, device)
    means_np = means_np[local_point_indices]
    colors_np = colors_np[local_point_indices]
    scales0 = scales0_full[torch.as_tensor(local_point_indices, dtype=torch.long, device=device)]

    from gsplat.strategy import DefaultStrategy

    means0 = torch.as_tensor(means_np, dtype=torch.float32, device=device)
    rgbs0 = torch.as_tensor(colors_np, dtype=torch.float32, device=device)
    colors0 = torch.zeros((means0.shape[0], (config.sh_degree + 1) ** 2, 3), dtype=torch.float32, device=device)
    colors0[:, 0, :] = _rgb_to_sh(rgbs0)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means0),
            "scales": torch.nn.Parameter(scales0),
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
    strategy = DefaultStrategy(
        prune_opa=config.prune_opa,
        grow_grad2d=config.grow_grad2d,
        refine_stop_iter=min(config.refine_stop_iter, config.steps),
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    if world_rank == 0:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = config.output_dir / "renders"
    train_render_dir = config.output_dir / "renders_train"
    if world_rank == 0:
        render_dir.mkdir(exist_ok=True)
        train_render_dir.mkdir(exist_ok=True)
        for stale_render in [*render_dir.glob("*.png"), *train_render_dir.glob("*.png")]:
            stale_render.unlink()
    stats_dir = config.output_dir / "stats"
    if world_rank == 0:
        stats_dir.mkdir(exist_ok=True)
    _distributed_barrier(world_size)

    view_order: list[int] = []
    pbar = tqdm(range(config.steps), desc="Training Gaussians", disable=world_rank != 0)
    for step in pbar:
        if not view_order:
            permutation = torch.randperm(len(train_indices)).tolist()
            view_order = [train_indices[i] for i in permutation]
        view_idx = view_order.pop()
        sh_degree_to_use = min(step // max(config.sh_degree_interval, 1), config.sh_degree)
        render_outputs, _alpha, info = _render(
            params,
            cameras[view_idx],
            config,
            sh_degree_to_use,
            render_mode="RGB+ED" if config.depth_loss else "RGB",
            distributed=is_distributed,
        )
        render = render_outputs[..., :3]
        if config.random_background:
            background = torch.rand((1, 1, 3), dtype=render.dtype, device=render.device)
            render = render + background * (1.0 - _alpha)
        target = targets[view_idx]
        target_mask = cameras[view_idx].get("mask") if config.mask_loss else None
        l1loss = _masked_l1_loss(render, target, target_mask)
        ssimloss = _masked_ssim_loss(render, target, target_mask)
        loss = torch.lerp(l1loss, ssimloss, config.ssim_lambda)
        mask_alpha_loss = None
        if target_mask is not None and config.mask_alpha_lambda > 0:
            mask_alpha_loss = _background_alpha_loss(_alpha, target_mask)
            loss = loss + config.mask_alpha_lambda * mask_alpha_loss
        depthloss = None
        if config.depth_loss and render_outputs.shape[-1] > 3:
            depthloss = _sparse_depth_loss(
                render_outputs[..., 3:4],
                cameras[view_idx]["depth_points"],
                cameras[view_idx]["depths"],
                scene_scale,
                config.depth_sample_count,
                config.depth_loss_clamp,
            )
            if depthloss is not None:
                loss = loss + config.depth_lambda * depthloss
        if config.opacity_reg > 0:
            loss = loss + config.opacity_reg * torch.sigmoid(params["opacities"]).mean()
        if config.scale_reg > 0:
            loss = loss + config.scale_reg * torch.exp(params["scales"]).mean()

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
        _cap_gaussians(params, optimizers, strategy_state, _local_shard_limit(config.max_gaussians, world_size, world_rank))
        if config.prune_every > 0 and step_num % config.prune_every == 0:
            _prune_floaters(params, optimizers, strategy_state, config.prune_opa, config.prune_large_scale)
        if (
            config.visibility_prune_every > 0
            and step_num >= config.visibility_prune_start
            and step_num % config.visibility_prune_every == 0
        ):
            _prune_unstable_gaussians(
                params,
                optimizers,
                strategy_state,
                cameras,
                train_indices,
                config,
                scene_scale,
                distributed=is_distributed,
            )
        if _should_reset_opacity(step_num, config.steps, config.opacity_reset_every, config.opacity_reset_until):
            _reset_opacities(params, optimizers, strategy_state, config.prune_opa * 2.0)
        if step_num == config.steps:
            final_removed = _final_prune(
                params,
                optimizers,
                strategy_state,
                cameras,
                train_indices,
                config,
                scene_scale,
                distributed=is_distributed,
            )
            if final_removed and world_rank == 0:
                print(f"Final cleanup removed {final_removed} Gaussians before saving.")
        if world_rank == 0 and (step == 0 or step_num % 50 == 0):
            pbar.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                l1=f"{float(l1loss.detach()):.4f}",
                ssim=f"{float(ssimloss.detach()):.4f}",
                sh=sh_degree_to_use,
                image=images[view_idx].name,
                gaussians=_global_count(len(params["means"]), device, world_size),
                depth="-" if depthloss is None else f"{float(depthloss.detach()):.4f}",
            )
        if step_num % config.save_every == 0 or step_num == config.steps:
            _save_checkpoint(config.output_dir / "checkpoint.pt", step_num, params, world_rank, world_size)
            num_gaussians = _global_count(len(params["means"]), device, world_size)
            opacity_mean = _global_tensor_mean(torch.sigmoid(params["opacities"].detach()), world_size)
            scale_mean = _global_tensor_mean(torch.exp(params["scales"].detach()), world_size)
            if world_rank == 0:
                _write_stats(
                    stats_dir / f"train_step_{step_num:06d}.json",
                    step_num,
                    float(loss.detach()),
                    float(l1loss.detach()),
                    float(ssimloss.detach()),
                    sh_degree_to_use,
                    images[view_idx].name,
                    num_gaussians,
                    opacity_mean=opacity_mean,
                scale_mean=scale_mean,
                train_views=len(train_indices),
                val_views=len(val_indices),
                depth_loss=None if depthloss is None else float(depthloss.detach()),
                mask_alpha_loss=None if mask_alpha_loss is None else float(mask_alpha_loss.detach()),
            )
        if step_num % config.render_every == 0 or step_num == config.steps:
            _write_renders(
                params,
                cameras,
                images,
                val_indices,
                config,
                render_dir,
                world_rank=world_rank,
                distributed=is_distributed,
            )
            _write_renders(
                params,
                cameras,
                images,
                train_indices[: min(12, len(train_indices))],
                config,
                train_render_dir,
                world_rank=world_rank,
                distributed=is_distributed,
            )
    _distributed_barrier(world_size)


def _load_training_views(
    model,
    image_dir: Path,
    scale: float,
    device: torch.device,
    allowed_depth_point_ids: set[int] | None = None,
    mask_dir: Path | None = None,
    mask_threshold: float = 0.5,
):
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
        mask = None
        mask_path = _find_mask_path(mask_dir, image.name) if mask_dir is not None else None
        if mask_path is not None:
            mask = _load_mask(mask_path, scale, device, mask_threshold)
            if mask.shape[:2] != (height, width):
                raise ValueError(
                    f"Mask shape mismatch for {image.name}: image {(height, width)}, mask {tuple(mask.shape[:2])}"
                )
        k = torch.as_tensor(camera_matrix(camera), dtype=torch.float32, device=device)
        k[0, :] *= width / camera.width
        k[1, :] *= height / camera.height
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = qvec_to_rotmat(image.qvec).astype(np.float32)
        viewmat[:3, 3] = image.tvec.astype(np.float32)
        camtoworld = np.linalg.inv(viewmat)
        depth_points, depths = _image_sparse_depths(
            model,
            image,
            width / camera.width,
            height / camera.height,
            allowed_point_ids=allowed_depth_point_ids,
        )
        cameras.append(
            {
                "K": k,
                "camtoworld": torch.as_tensor(camtoworld, dtype=torch.float32, device=device),
                "width": width,
                "height": height,
                "mask": mask,
                "depth_points": torch.as_tensor(depth_points, dtype=torch.float32, device=device),
                "depths": torch.as_tensor(depths, dtype=torch.float32, device=device),
            }
        )
        images.append(image)
        targets.append(target)
    if not targets:
        raise ValueError(f"No training images found under {image_dir}")
    return cameras, images, targets


def _image_sparse_depths(
    model,
    image,
    scale_x: float,
    scale_y: float,
    allowed_point_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if image.xys.size == 0 or image.point3d_ids.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    rot = qvec_to_rotmat(image.qvec)
    points_xy = []
    depths = []
    width_limit = model.cameras[image.camera_id].width * scale_x
    height_limit = model.cameras[image.camera_id].height * scale_y
    for xy, point_id in zip(image.xys, image.point3d_ids):
        point_id = int(point_id)
        if allowed_point_ids is not None and point_id not in allowed_point_ids:
            continue
        point = model.points3d.get(int(point_id))
        if point is None:
            continue
        depth = float((rot @ point.xyz + image.tvec)[2])
        x = float(xy[0] * scale_x)
        y = float(xy[1] * scale_y)
        if depth <= 0.0 or not np.isfinite(depth):
            continue
        if x < 0.0 or y < 0.0 or x >= width_limit or y >= height_limit:
            continue
        points_xy.append((x, y))
        depths.append(depth)
    if not points_xy:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    points_arr = np.asarray(points_xy, dtype=np.float32)
    depths_arr = np.asarray(depths, dtype=np.float32)
    keep = _robust_depth_mask(depths_arr)
    return points_arr[keep], depths_arr[keep]


def _split_view_indices(count: int, test_every: int) -> tuple[list[int], list[int]]:
    if count <= 0:
        return [], []
    if test_every <= 1:
        return list(range(count)), []
    val_indices = [idx for idx in range(count) if idx % test_every == 0]
    train_indices = [idx for idx in range(count) if idx % test_every != 0]
    if not train_indices:
        train_indices = list(range(count))
        val_indices = []
    return train_indices, val_indices


def _initial_points(model, max_points: int, min_track_len: int = 2) -> tuple[np.ndarray, np.ndarray]:
    return _points_to_arrays(_select_initial_points(model, max_points, min_track_len))


def _select_initial_points(model, max_points: int, min_track_len: int = 2) -> list:
    points = [p for p in model.points3d.values() if np.all(np.isfinite(p.xyz)) and len(p.track) >= min_track_len]
    points = _filter_point_outliers(points)
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = [points[i] for i in keep]
    return points


def _points_to_arrays(points: list) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    means = np.stack([p.xyz for p in points], axis=0).astype(np.float32)
    colors = np.stack([p.rgb.astype(np.float32) / 255.0 for p in points], axis=0).astype(np.float32)
    return means, colors


def _robust_depth_mask(depths: np.ndarray) -> np.ndarray:
    depths = np.asarray(depths, dtype=np.float32)
    valid = np.isfinite(depths) & (depths > 0.0)
    if int(valid.sum()) < 32:
        return valid

    valid_depths = depths[valid].astype(np.float64)
    median = float(np.median(valid_depths))
    if not np.isfinite(median) or median <= 0.0:
        return valid

    q01, q99 = np.quantile(valid_depths, [0.01, 0.99])
    lower = max(float(q01), median * 0.2)
    upper = min(float(q99), median * 5.0)
    if lower >= upper:
        lower = float(q01)
        upper = float(q99)
    return valid & (depths >= lower) & (depths <= upper)


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


def _render(
    params,
    camera,
    config: TrainConfig,
    sh_degree: int | None = None,
    render_mode: str = "RGB",
    distributed: bool = False,
):
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
        render_mode=render_mode,
        distributed=distributed,
    )
    return colors[0], alpha[0], info


def _sparse_depth_loss(
    rendered_depth: torch.Tensor,
    points: torch.Tensor,
    depths: torch.Tensor,
    scene_scale: float,
    sample_count: int,
    loss_clamp: float = 2.0,
) -> torch.Tensor | None:
    if points.numel() == 0 or depths.numel() == 0:
        return None
    if sample_count > 0 and len(points) > sample_count:
        indices = torch.randperm(len(points), device=points.device)[:sample_count]
        points = points[indices]
        depths = depths[indices]

    height, width = rendered_depth.shape[:2]
    if width <= 1 or height <= 1:
        return None
    grid = points.clone()
    grid[:, 0] = grid[:, 0] / (width - 1) * 2.0 - 1.0
    grid[:, 1] = grid[:, 1] / (height - 1) * 2.0 - 1.0
    grid = grid.view(1, -1, 1, 2)
    sampled = F.grid_sample(
        rendered_depth.permute(2, 0, 1).unsqueeze(0),
        grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="zeros",
    ).view(-1)
    valid = (sampled > 0.0) & (depths > 0.0)
    if not bool(valid.any()):
        return None
    pred_disp = 1.0 / sampled[valid].clamp_min(1e-6)
    gt_disp = 1.0 / depths[valid].clamp_min(1e-6)
    residual = (pred_disp - gt_disp).abs() * scene_scale
    if loss_clamp > 0:
        residual = residual.clamp(max=loss_clamp)
    return residual.mean()


def _save_checkpoint(path: Path, step: int, params, world_rank: int = 0, world_size: int = 1) -> None:
    output_params = _gather_params_for_output(params, world_size)
    if world_rank != 0:
        return
    torch.save(
        {
            "step": step,
            "splats": {key: value.detach().cpu() for key, value in output_params.items()},
            "means": output_params["means"].detach().cpu(),
            "quats": output_params["quats"].detach().cpu(),
            "scales": torch.exp(output_params["scales"].detach()).cpu(),
            "opacities": torch.sigmoid(output_params["opacities"].detach()).cpu(),
            "sh0": output_params["sh0"].detach().cpu(),
            "shN": output_params["shN"].detach().cpu(),
            "colors": _sh_to_rgb(output_params["sh0"].detach()[:, 0, :]).clamp(0.0, 1.0).cpu(),
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


def _find_mask_path(mask_dir: Path | None, image_name: str) -> Path | None:
    if mask_dir is None:
        return None
    image_path = Path(image_name)
    stem = image_path.stem
    suffixes = [image_path.suffix] if image_path.suffix else []
    suffixes.extend([".png", ".jpg", ".jpeg"])

    candidate_stems = [stem]
    if stem.startswith("rgb_"):
        candidate_stems.append("msk_" + stem[len("rgb_") :])
    candidate_stems.extend([f"mask_{stem}", f"msk_{stem}"])

    seen: set[Path] = set()
    for candidate_stem in candidate_stems:
        for suffix in suffixes:
            candidate = mask_dir / f"{candidate_stem}{suffix}"
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
    return None


def _load_mask(path: Path, scale: float, device: torch.device, threshold: float = 0.5) -> torch.Tensor:
    with PILImage.open(path) as im:
        im = im.convert("L")
        if scale != 1.0:
            width = max(1, int(round(im.width * scale)))
            height = max(1, int(round(im.height * scale)))
            im = im.resize((width, height), PILImage.Resampling.NEAREST)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    mask = (arr >= threshold).astype(np.float32)[..., None]
    return torch.as_tensor(mask, dtype=torch.float32, device=device)


def _masked_l1_loss(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    residual = (render - target).abs()
    if mask is None:
        return residual.mean()
    denom = mask.sum() * render.shape[-1]
    if float(denom.detach().cpu()) <= 0.0:
        return residual.mean()
    return (residual * mask).sum() / denom.clamp_min(1.0)


def _masked_ssim_loss(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return _ssim_loss(render[None].permute(0, 3, 1, 2), target[None].permute(0, 3, 1, 2))
    masked_render = render * mask
    masked_target = target * mask
    return _ssim_loss(masked_render[None].permute(0, 3, 1, 2), masked_target[None].permute(0, 3, 1, 2))


def _background_alpha_loss(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    background = 1.0 - mask
    denom = background.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return alpha.new_zeros(())
    return (alpha * background).sum() / denom.clamp_min(1.0)


def _write_render(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    PILImage.fromarray(out).save(path.with_suffix(".png"))


@torch.no_grad()
def _write_renders(
    params,
    cameras,
    images,
    indices: list[int],
    config: TrainConfig,
    render_dir: Path,
    world_rank: int = 0,
    distributed: bool = False,
) -> None:
    indices = indices if indices else list(range(len(images)))
    sh_degree_to_use = config.sh_degree
    for idx in indices:
        render, _alpha, _info = _render(params, cameras[idx], config, sh_degree_to_use, distributed=distributed)
        if world_rank == 0:
            _write_render(render_dir / images[idx].name, render.detach())


@torch.no_grad()
def _cap_gaussians(params, optimizers, strategy_state, max_gaussians: int) -> None:
    if max_gaussians <= 0 or len(params["means"]) <= max_gaussians:
        return
    from gsplat.strategy.ops import remove

    opacities = torch.sigmoid(params["opacities"].detach())
    remove_count = len(params["means"]) - max_gaussians
    remove_ids = torch.topk(opacities, k=remove_count, largest=False).indices
    mask = torch.zeros_like(opacities, dtype=torch.bool)
    mask[remove_ids] = True
    remove(params, optimizers, strategy_state, mask)


@torch.no_grad()
def _prune_floaters(params, optimizers, strategy_state, min_opacity: float, max_scale: float) -> int:
    from gsplat.strategy.ops import remove

    if len(params["means"]) <= 1:
        return 0
    opacities = torch.sigmoid(params["opacities"].detach())
    scales = torch.exp(params["scales"].detach()).max(dim=-1).values
    mask = opacities < min_opacity
    if max_scale > 0:
        mask = mask | (scales > max_scale)
    if bool(mask.all()):
        mask[torch.argmax(opacities)] = False
    removed = int(mask.sum().item())
    if bool(mask.any()):
        remove(params, optimizers, strategy_state, mask)
    return removed


@torch.no_grad()
def _prune_unstable_gaussians(
    params,
    optimizers,
    strategy_state,
    cameras: list[dict],
    view_indices: list[int],
    config: TrainConfig,
    scene_scale: float,
    distributed: bool = False,
) -> int:
    from gsplat.strategy.ops import remove

    count = len(params["means"])
    if count <= 1:
        return 0

    device = params["means"].device
    remove_mask = torch.zeros(count, dtype=torch.bool, device=device)
    min_views = min(max(config.visibility_min_views, 0), len(view_indices))
    if min_views > 0 and view_indices:
        visible_counts = torch.zeros(count, dtype=torch.int16, device=device)
        for view_idx in view_indices:
            _rendered, _alpha, info = _render(
                params,
                cameras[view_idx],
                config,
                config.sh_degree,
                distributed=distributed,
            )
            visible = _visible_gaussians_from_info(info, count, device)
            if visible is not None:
                visible_counts += visible.to(torch.int16)
        remove_mask |= visible_counts < min_views

    if config.prune_scene_radius > 0 and scene_scale > 0:
        radius = torch.linalg.norm(params["means"].detach(), dim=-1)
        remove_mask |= radius > (scene_scale * config.prune_scene_radius)

    if bool(remove_mask.all()):
        opacities = torch.sigmoid(params["opacities"].detach())
        remove_mask[torch.argmax(opacities)] = False
    removed = int(remove_mask.sum().item())
    if removed:
        remove(params, optimizers, strategy_state, remove_mask)
        print(f"Pruned {removed} unstable Gaussians; kept {len(params['means'])}.")
    return removed


@torch.no_grad()
def _final_prune(
    params,
    optimizers,
    strategy_state,
    cameras: list[dict],
    view_indices: list[int],
    config: TrainConfig,
    scene_scale: float,
    distributed: bool = False,
) -> int:
    removed = 0
    if config.final_prune_opa > 0 or config.final_prune_large_scale > 0:
        removed += _prune_floaters(
            params,
            optimizers,
            strategy_state,
            config.final_prune_opa,
            config.final_prune_large_scale,
        )
    if (
        config.final_visibility_min_views > 0
        or (config.final_prune_scene_radius > 0 and scene_scale > 0)
    ):
        cleanup_config = TrainConfig(
            scene=config.scene,
            sparse_dir=config.sparse_dir,
            image_dir=config.image_dir,
            output_dir=config.output_dir,
            visibility_min_views=config.final_visibility_min_views,
            prune_scene_radius=config.final_prune_scene_radius,
            sh_degree=config.sh_degree,
            sh_degree_interval=config.sh_degree_interval,
            packed=config.packed,
            antialiased=config.antialiased,
            near_plane=config.near_plane,
            far_plane=config.far_plane,
        )
        removed += _prune_unstable_gaussians(
            params,
            optimizers,
            strategy_state,
            cameras,
            view_indices,
            cleanup_config,
            scene_scale,
            distributed=distributed,
        )
    return removed


def _should_reset_opacity(step_num: int, total_steps: int, reset_every: int, reset_until: int) -> bool:
    if reset_every <= 0 or step_num <= 0 or step_num >= total_steps:
        return False
    effective_until = reset_until if reset_until > 0 else total_steps
    return step_num <= effective_until and step_num % reset_every == 0


@torch.no_grad()
def _reset_opacities(params, optimizers, strategy_state, value: float) -> None:
    from gsplat.strategy.ops import reset_opa

    reset_opa(params=params, optimizers=optimizers, state=strategy_state, value=value)


def _local_shard_limit(total_limit: int, world_size: int, world_rank: int) -> int:
    if total_limit <= 0 or world_size <= 1:
        return total_limit
    base = total_limit // world_size
    remainder = total_limit % world_size
    return base + (1 if world_rank < remainder else 0)


def _distributed_barrier(world_size: int) -> None:
    if world_size <= 1:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _global_count(local_count: int, device: torch.device, world_size: int) -> int:
    if world_size <= 1:
        return int(local_count)
    import torch.distributed as dist

    count = torch.tensor([local_count], dtype=torch.long, device=device)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return int(count.item())


def _global_tensor_mean(values: torch.Tensor, world_size: int) -> float:
    if values.numel() == 0:
        device = values.device
        local = torch.zeros(2, dtype=torch.float64, device=device)
    else:
        local = torch.tensor(
            [float(values.sum().detach()), float(values.numel())],
            dtype=torch.float64,
            device=values.device,
        )
    if world_size > 1:
        import torch.distributed as dist

        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    if local[1].item() == 0:
        return 0.0
    return float((local[0] / local[1]).item())


def _gather_params_for_output(params, world_size: int) -> dict[str, torch.Tensor]:
    if world_size <= 1:
        return {key: value.detach() for key, value in params.items()}
    return {key: _all_gather_variable_first_dim(value.detach(), world_size) for key, value in params.items()}


def _all_gather_variable_first_dim(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    import torch.distributed as dist

    local_n = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    lengths = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(lengths, local_n)
    lengths_int = [int(length.item()) for length in lengths]
    max_n = max(lengths_int)
    if tensor.shape[0] < max_n:
        pad_shape = (max_n - tensor.shape[0], *tensor.shape[1:])
        tensor_to_gather = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)
    else:
        tensor_to_gather = tensor
    gathered = [torch.empty_like(tensor_to_gather) for _ in range(world_size)]
    dist.all_gather(gathered, tensor_to_gather)
    return torch.cat([item[:length] for item, length in zip(gathered, lengths_int)], dim=0)


def _visible_gaussians_from_info(info: dict, count: int, device: torch.device) -> torch.Tensor | None:
    radii = info.get("radii")
    if radii is not None:
        radii = radii.to(device)
        if radii.ndim == 1 and radii.shape[0] == count:
            return radii > 0
        if radii.shape[-1] == count:
            return (radii > 0).reshape(-1, count).any(dim=0)
        if radii.ndim >= 2 and radii.shape[-2] == count:
            return (radii > 0).all(dim=-1).reshape(-1, count).any(dim=0)

    gaussian_ids = info.get("gaussian_ids")
    if gaussian_ids is None:
        return None
    visible = torch.zeros(count, dtype=torch.bool, device=device)
    gaussian_ids = gaussian_ids.to(device=device, dtype=torch.long)
    gaussian_ids = gaussian_ids[(gaussian_ids >= 0) & (gaussian_ids < count)]
    if gaussian_ids.numel() > 0:
        visible[gaussian_ids] = True
    return visible


def _write_stats(
    path: Path,
    step: int,
    loss: float,
    l1loss: float,
    ssimloss: float,
    sh_degree: int,
    image_name: str,
    num_gaussians: int,
    opacity_mean: float | None = None,
    scale_mean: float | None = None,
    train_views: int | None = None,
    val_views: int | None = None,
    depth_loss: float | None = None,
    mask_alpha_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        payload = {
            "step": step,
            "loss": loss,
            "l1": l1loss,
            "ssim": ssimloss,
            "sh_degree": sh_degree,
            "image": image_name,
            "num_gaussians": num_gaussians,
        }
        if opacity_mean is not None:
            payload["opacity_mean"] = opacity_mean
        if scale_mean is not None:
            payload["scale_mean"] = scale_mean
        if train_views is not None:
            payload["train_views"] = train_views
        if val_views is not None:
            payload["val_views"] = val_views
        if depth_loss is not None:
            payload["depth_loss"] = depth_loss
        if mask_alpha_loss is not None:
            payload["mask_alpha_loss"] = mask_alpha_loss
        json.dump(payload, f, indent=2)


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
