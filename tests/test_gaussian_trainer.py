from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import torch
from PIL import Image as PILImage

from vggt_gaussian_reconstruction.colmap import Camera, Image, Point3D, Reconstruction, write_model
from vggt_gaussian_reconstruction.gaussian_trainer import (
    TrainConfig,
    _initial_points,
    _load_training_views,
    _prepare_optimizer_grads,
    _render,
)
from vggt_gaussian_reconstruction.geometry import rotmat_to_qvec


def test_gaussian_trainer_loads_colmap_views(tmp_path: Path):
    sparse = tmp_path / "sparse" / "0"
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    PILImage.fromarray(np.full((6, 8, 3), 128, dtype=np.uint8)).save(images_dir / "im1.png")

    camera = Camera(1, "PINHOLE", 8, 6, np.array([4.0, 4.0, 4.0, 3.0]))
    image = Image(1, rotmat_to_qvec(np.eye(3)), np.zeros(3), 1, "im1.png")
    point = Point3D(1, np.array([0.0, 0.0, 2.0]), np.array([255, 128, 0], dtype=np.uint8), 0.0, [(1, 0), (1, 1)])
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


def test_initial_points_filters_extreme_outliers():
    camera = Camera(1, "PINHOLE", 8, 6, np.array([4.0, 4.0, 4.0, 3.0]))
    image = Image(1, rotmat_to_qvec(np.eye(3)), np.zeros(3), 1, "im1.png")
    points = {
        i: Point3D(
            i,
            np.array([float(i % 10) / 100.0, float(i // 10) / 100.0, 1.0]),
            np.array([255, 128, 0], dtype=np.uint8),
            0.0,
            [(1, 0), (1, 1)],
        )
        for i in range(200)
    }
    points[999] = Point3D(
        999,
        np.array([10_000.0, 0.0, 0.0]),
        np.array([0, 0, 0], dtype=np.uint8),
        0.0,
        [(1, 0), (1, 1)],
    )
    model = Reconstruction({1: camera}, {1: image}, points)

    means, _colors = _initial_points(model, 1000)

    assert means.shape[0] < len(points)
    assert np.max(np.linalg.norm(means, axis=1)) < 10.0


def test_render_passes_simple_trainer_rasterization_args(monkeypatch):
    captured = {}

    def fake_rasterization(**kwargs):
        captured["has_backgrounds"] = "backgrounds" in kwargs
        captured["colors_shape"] = tuple(kwargs["colors"].shape)
        captured["viewmats"] = kwargs["viewmats"].clone()
        captured["sh_degree"] = kwargs["sh_degree"]
        captured["near_plane"] = kwargs["near_plane"]
        captured["far_plane"] = kwargs["far_plane"]
        captured["packed"] = kwargs["packed"]
        captured["rasterize_mode"] = kwargs["rasterize_mode"]
        return torch.ones((1, 2, 3, 3), dtype=kwargs["means"].dtype), torch.ones((1, 2, 3, 1)), {}

    fake_gsplat = types.ModuleType("gsplat")
    fake_rendering = types.ModuleType("gsplat.rendering")
    fake_rendering.rasterization = fake_rasterization
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", fake_rendering)

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros((1, 3), dtype=torch.float32)),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)),
            "scales": torch.nn.Parameter(torch.zeros((1, 3), dtype=torch.float32)),
            "opacities": torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32)),
            "sh0": torch.nn.Parameter(torch.zeros((1, 1, 3), dtype=torch.float32)),
            "shN": torch.nn.Parameter(torch.zeros((1, 15, 3), dtype=torch.float32)),
        }
    )
    camera = {
        "K": torch.eye(3, dtype=torch.float32),
        "camtoworld": torch.eye(4, dtype=torch.float32),
        "width": 3,
        "height": 2,
    }

    render, alpha, info = _render(params, camera, TrainConfig(Path("."), Path("."), Path("."), Path(".")), sh_degree=2)

    assert captured["has_backgrounds"] is False
    assert captured["colors_shape"] == (1, 16, 3)
    assert torch.allclose(captured["viewmats"], torch.eye(4)[None])
    assert captured["sh_degree"] == 2
    assert captured["near_plane"] == 0.01
    assert captured["far_plane"] == 1e10
    assert captured["packed"] is False
    assert captured["rasterize_mode"] == "classic"
    assert render.shape == (2, 3, 3)
    assert alpha.shape == (2, 3, 1)
    assert info == {}


def test_prepare_optimizer_grads_matches_parameter_dtype_and_layout():
    param = torch.nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    param.grad = torch.ones((3, 2), dtype=torch.float32).t()
    optimizer = torch.optim.Adam([{"params": param, "lr": 1e-3, "name": "means"}])

    _prepare_optimizer_grads(optimizer)

    assert param.grad is not None
    assert param.grad.dtype == param.dtype
    assert param.grad.device == param.device
    assert param.grad.layout == param.layout
    assert param.grad.is_contiguous()
