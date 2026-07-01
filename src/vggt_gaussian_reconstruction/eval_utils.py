from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .colmap import read_model


def reconstruction_summary(sparse_dir: Path) -> dict:
    model = read_model(sparse_dir)
    obs_count = 0
    errors = []
    for point in model.points3d.values():
        obs_count += len(point.track)
        if math.isfinite(point.error):
            errors.append(float(point.error))
    return {
        "cameras": len(model.cameras),
        "images": len(model.images),
        "points3d": len(model.points3d),
        "observations": obs_count,
        "mean_point_error": float(np.mean(errors)) if errors else None,
    }


def image_metrics(render_dir: Path, gt_dir: Path) -> dict:
    pairs = []
    for pred in sorted(render_dir.glob("*.png")):
        gt = gt_dir / pred.name
        if gt.exists():
            pairs.append((pred, gt))
    values = []
    for pred, gt in pairs:
        p = _load_rgb(pred)
        g = _load_rgb(gt)
        h = min(p.shape[0], g.shape[0])
        w = min(p.shape[1], g.shape[1])
        p = p[:h, :w]
        g = g[:h, :w]
        mse = float(np.mean((p - g) ** 2))
        psnr = 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)
        values.append({"image": pred.name, "mse": mse, "psnr": psnr})
    return {
        "pairs": len(values),
        "mean_psnr": float(np.mean([v["psnr"] for v in values])) if values else None,
        "per_image": values,
    }


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
