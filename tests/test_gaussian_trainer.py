from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import torch
from PIL import Image as PILImage

from vggt_gaussian_reconstruction.colmap import Camera, Image, Point3D, Reconstruction, write_model
from vggt_gaussian_reconstruction.gaussian_trainer import _initial_points, _load_training_views, _render
from vggt_gaussian_reconstruction.geometry import rotmat_to_qvec


def test_gaussian_trainer_loads_colmap_views(tmp_path: Path):
    sparse = tmp_path / "sparse" / "0"
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    PILImage.fromarray(np.full((6, 8, 3), 128, dtype=np.uint8)).save(images_dir / "im1.png")

    camera = Camera(1, "PINHOLE", 8, 6, np.array([4.0, 4.0, 4.0, 3.0]))
    image = Image(1, rotmat_to_qvec(np.eye(3)), np.zeros(3), 1, "im1.png")
    point = Point3D(1, np.array([0.0, 0.0, 2.0]), np.array([255, 128, 0], dtype=np.uint8), 0.0, [])
    write_model(Reconstruction({1: camera}, {1: image}, {1: point}), sparse)

    model = Reconstruction({1: camera}, {1: image}, {1: point})
    cameras, images, targets = _load_training_views(model, images_dir, 0.5, torch.device("cpu"))
    means, colors = _initial_points(model, 100)

    assert images[0].name == "im1.png"
    assert cameras[0]["width"] == 4
    assert cameras[0]["height"] == 3
    assert targets[0].shape == (3, 4, 3)
    assert means.shape == (1, 3)
    assert np.allclose(colors, [[1.0, 128.0 / 255.0, 0.0]])


def test_render_passes_packed_background_shape(monkeypatch):
    captured = {}

    def fake_rasterization(**kwargs):
        captured["backgrounds_shape"] = tuple(kwargs["backgrounds"].shape)
        return torch.ones((1, 2, 3, 3), dtype=kwargs["means"].dtype), None, None

    fake_gsplat = types.ModuleType("gsplat")
    fake_rendering = types.ModuleType("gsplat.rendering")
    fake_rendering.rasterization = fake_rasterization
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", fake_rendering)

    means = torch.zeros((1, 3), dtype=torch.float32)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    log_scales = torch.zeros((1, 3), dtype=torch.float32)
    opacity_logits = torch.zeros((1,), dtype=torch.float32)
    rgb_logits = torch.zeros((1, 3), dtype=torch.float32)
    camera = {
        "K": torch.eye(3, dtype=torch.float32),
        "viewmat": torch.eye(4, dtype=torch.float32),
        "width": 3,
        "height": 2,
    }

    render = _render(means, quats, log_scales, opacity_logits, rgb_logits, camera, None)

    assert captured["backgrounds_shape"] == (3,)
    assert render.shape == (2, 3, 3)
