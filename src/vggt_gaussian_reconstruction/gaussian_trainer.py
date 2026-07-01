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
from .geometry import qvec_to_rotmat


@dataclass
class TrainConfig:
    scene: Path
    sparse_dir: Path
    image_dir: Path
    output_dir: Path
    steps: int = 7000
    device: str = "cuda"
    lr: float = 1e-2
    image_scale: float = 1.0
    max_points: int = 200_000
    save_every: int = 1000
    render_every: int = 1000


def train_gaussians(config: TrainConfig) -> None:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model = read_model(config.sparse_dir)
    cameras, images, targets = _load_training_views(model, config.image_dir, config.image_scale, device)
    means_np, colors_np = _initial_points(model, config.max_points)

    if len(means_np) == 0:
        raise ValueError(f"No 3D points found in {config.sparse_dir}")

    means = torch.nn.Parameter(torch.as_tensor(means_np, dtype=torch.float32, device=device))
    rgb_logits = torch.nn.Parameter(_safe_logit(torch.as_tensor(colors_np, dtype=torch.float32, device=device)))
    quats = torch.nn.Parameter(torch.zeros((means.shape[0], 4), dtype=torch.float32, device=device))
    quats.data[:, 0] = 1.0
    log_scales = torch.nn.Parameter(torch.full((means.shape[0], 3), math.log(_initial_scale(means_np)), dtype=torch.float32, device=device))
    opacity_logits = torch.nn.Parameter(torch.full((means.shape[0],), _logit(0.35), dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": config.lr},
            {"params": [rgb_logits], "lr": config.lr},
            {"params": [log_scales], "lr": config.lr * 0.5},
            {"params": [opacity_logits], "lr": config.lr * 0.5},
            {"params": [quats], "lr": config.lr * 0.25},
        ]
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = config.output_dir / "renders"
    render_dir.mkdir(exist_ok=True)

    pbar = tqdm(range(1, config.steps + 1), desc="Training Gaussians")
    for step in pbar:
        view_idx = (step - 1) % len(images)
        render = _render(
            means,
            quats,
            log_scales,
            opacity_logits,
            rgb_logits,
            cameras[view_idx],
            images[view_idx],
        )
        target = targets[view_idx]
        loss = F.l1_loss(render, target) + 0.2 * F.mse_loss(render, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            quats[:] = F.normalize(quats, dim=-1)

        if step == 1 or step % 50 == 0:
            pbar.set_postfix(loss=f"{float(loss.detach()):.4f}", image=images[view_idx].name)
        if step % config.save_every == 0 or step == config.steps:
            _save_checkpoint(config.output_dir / "checkpoint.pt", step, means, quats, log_scales, opacity_logits, rgb_logits)
        if step % config.render_every == 0 or step == config.steps:
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
        cameras.append(
            {
                "K": k,
                "viewmat": torch.as_tensor(viewmat, dtype=torch.float32, device=device),
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
    points = list(model.points3d.values())
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = [points[i] for i in keep]
    means = np.stack([p.xyz for p in points], axis=0).astype(np.float32)
    colors = np.stack([p.rgb.astype(np.float32) / 255.0 for p in points], axis=0).astype(np.float32)
    return means, colors


def _render(means, quats, log_scales, opacity_logits, rgb_logits, camera, _image):
    from gsplat.rendering import rasterization

    colors, _, _ = rasterization(
        means=means,
        quats=F.normalize(quats, dim=-1),
        scales=torch.exp(log_scales),
        opacities=torch.sigmoid(opacity_logits),
        colors=torch.sigmoid(rgb_logits),
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=camera["width"],
        height=camera["height"],
        packed=True,
        backgrounds=torch.ones((3,), dtype=means.dtype, device=means.device),
    )
    return colors[0, ..., :3].clamp(0.0, 1.0)


def _save_checkpoint(path: Path, step: int, means, quats, log_scales, opacity_logits, rgb_logits) -> None:
    torch.save(
        {
            "step": step,
            "means": means.detach().cpu(),
            "quats": F.normalize(quats.detach(), dim=-1).cpu(),
            "scales": torch.exp(log_scales.detach()).cpu(),
            "opacities": torch.sigmoid(opacity_logits.detach()).cpu(),
            "colors": torch.sigmoid(rgb_logits.detach()).cpu(),
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


def _safe_logit(x: torch.Tensor) -> torch.Tensor:
    return torch.logit(x.clamp(1e-4, 1.0 - 1e-4))


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _initial_scale(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.01
    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    return float(max(extent / math.sqrt(len(points)) * 0.1, 1e-4))
