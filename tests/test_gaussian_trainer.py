from __future__ import annotations

from pathlib import Path
import json
import sys
import types

import numpy as np
import torch
from PIL import Image as PILImage

from vggt_gaussian_reconstruction.colmap import Camera, Image, Point3D, Reconstruction, write_model
from vggt_gaussian_reconstruction.gaussian_trainer import (
    TrainConfig,
    _background_alpha_loss,
    _find_mask_path,
    _image_sparse_depths,
    _initial_points,
    _load_mask,
    _load_training_views,
    _masked_l1_loss,
    _prepare_optimizer_grads,
    _prune_unstable_gaussians,
    _render,
    _robust_depth_mask,
    _reset_opacities,
    _should_reset_opacity,
    _sparse_depth_loss,
    _split_view_indices,
    _train_gaussians_worker,
    _visible_gaussians_from_info,
    _write_stats,
    train_gaussians,
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


def test_gaussian_trainer_loads_matching_foreground_masks(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    PILImage.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(images_dir / "rgb_0000.png")
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 255
    PILImage.fromarray(mask).save(masks_dir / "msk_0000.png")

    camera = Camera(1, "PINHOLE", 4, 4, np.array([2.0, 2.0, 2.0, 2.0]))
    image = Image(1, rotmat_to_qvec(np.eye(3)), np.zeros(3), 1, "rgb_0000.png")
    model = Reconstruction({1: camera}, {1: image}, {})

    cameras, _images, _targets = _load_training_views(
        model,
        images_dir,
        0.5,
        torch.device("cpu"),
        mask_dir=masks_dir,
    )

    assert cameras[0]["mask"] is not None
    assert cameras[0]["mask"].shape == (2, 2, 1)
    assert float(cameras[0]["mask"].sum()) == 1.0
    assert _find_mask_path(masks_dir, "rgb_0000.png") == masks_dir / "msk_0000.png"


def test_masked_losses_ignore_background_rgb_and_penalize_background_alpha():
    render = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    target = torch.zeros_like(render)
    mask = torch.tensor([[[1.0], [0.0]], [[0.0], [0.0]]], dtype=torch.float32)
    alpha = torch.tensor([[[0.2], [0.5]], [[0.25], [0.0]]], dtype=torch.float32)

    assert torch.allclose(_masked_l1_loss(render, target, mask), torch.tensor(1.0 / 3.0))
    assert torch.allclose(_background_alpha_loss(alpha, mask), torch.tensor(0.25))


def test_load_mask_thresholds_and_resizes_with_nearest(tmp_path: Path):
    mask_path = tmp_path / "mask.png"
    PILImage.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(mask_path)

    mask = _load_mask(mask_path, 0.5, torch.device("cpu"), threshold=0.5)

    assert mask.shape == (1, 1, 1)
    assert float(mask.item()) == 0.0


def test_image_sparse_depths_uses_colmap_observations_and_scaled_pixels():
    camera = Camera(1, "PINHOLE", 8, 6, np.array([4.0, 4.0, 4.0, 3.0]))
    image = Image(
        1,
        rotmat_to_qvec(np.eye(3)),
        np.zeros(3),
        1,
        "im1.png",
        xys=np.array([[4.0, 3.0], [100.0, 100.0]], dtype=np.float64),
        point3d_ids=np.array([1, 2], dtype=np.int64),
    )
    points = {
        1: Point3D(1, np.array([0.0, 0.0, 2.0]), np.array([255, 128, 0], dtype=np.uint8), 0.0, [(1, 0)]),
        2: Point3D(2, np.array([0.0, 0.0, 3.0]), np.array([255, 128, 0], dtype=np.uint8), 0.0, [(1, 1)]),
    }
    model = Reconstruction({1: camera}, {1: image}, points)

    pixels, depths = _image_sparse_depths(model, image, scale_x=0.5, scale_y=0.5)

    assert pixels.shape == (1, 2)
    assert np.allclose(pixels[0], [2.0, 1.5])
    assert np.allclose(depths, [2.0])


def test_image_sparse_depths_respects_allowed_points_and_filters_depth_outliers():
    camera = Camera(1, "PINHOLE", 100, 80, np.array([50.0, 50.0, 50.0, 40.0]))
    xys = []
    point_ids = []
    points = {}
    for idx in range(50):
        point_id = idx + 1
        xys.append([10.0 + idx, 20.0])
        point_ids.append(point_id)
        points[point_id] = Point3D(
            point_id,
            np.array([0.0, 0.0, 1.0 + idx * 0.001]),
            np.array([255, 128, 0], dtype=np.uint8),
            0.0,
            [(1, idx)],
        )
    points[1000] = Point3D(1000, np.array([0.0, 0.0, 0.001]), np.array([0, 0, 0], dtype=np.uint8), 0.0, [(1, 50)])
    points[1001] = Point3D(1001, np.array([0.0, 0.0, 100.0]), np.array([0, 0, 0], dtype=np.uint8), 0.0, [(1, 51)])
    points[1002] = Point3D(1002, np.array([0.0, 0.0, 1.0]), np.array([0, 0, 0], dtype=np.uint8), 0.0, [(1, 52)])
    xys.extend([[70.0, 20.0], [71.0, 20.0], [72.0, 20.0]])
    point_ids.extend([1000, 1001, 1002])
    image = Image(
        1,
        rotmat_to_qvec(np.eye(3)),
        np.zeros(3),
        1,
        "im1.png",
        xys=np.asarray(xys, dtype=np.float64),
        point3d_ids=np.asarray(point_ids, dtype=np.int64),
    )
    model = Reconstruction({1: camera}, {1: image}, points)

    pixels, depths = _image_sparse_depths(model, image, 1.0, 1.0, allowed_point_ids=set(range(1, 1002)))

    assert 0.001 not in depths
    assert 100.0 not in depths
    assert len(depths) == len(pixels)
    assert len(depths) >= 48


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


def test_split_view_indices_matches_simple_trainer_test_every():
    train, val = _split_view_indices(10, 4)

    assert train == [1, 2, 3, 5, 6, 7, 9]
    assert val == [0, 4, 8]


def test_write_stats_json(tmp_path: Path):
    path = tmp_path / "stats" / "train_step_000010.json"

    _write_stats(path, 10, 0.5, 0.4, 0.3, 2, "frame_0001.png", 123)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["step"] == 10
    assert payload["num_gaussians"] == 123
    assert payload["image"] == "frame_0001.png"


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
        captured["render_mode"] = kwargs["render_mode"]
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
    assert captured["render_mode"] == "RGB"
    assert render.shape == (2, 3, 3)
    assert alpha.shape == (2, 3, 1)
    assert info == {}


def test_sparse_depth_loss_uses_disparity_space():
    rendered_depth = torch.full((2, 3, 1), 2.0, dtype=torch.float32)
    points = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    depths = torch.tensor([4.0], dtype=torch.float32)

    loss = _sparse_depth_loss(rendered_depth, points, depths, scene_scale=2.0, sample_count=0)

    assert loss is not None
    assert torch.allclose(loss, torch.tensor(0.5))


def test_sparse_depth_loss_clamps_large_residuals():
    rendered_depth = torch.full((2, 3, 1), 2.0, dtype=torch.float32)
    points = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    depths = torch.tensor([0.001], dtype=torch.float32)

    loss = _sparse_depth_loss(rendered_depth, points, depths, scene_scale=2.0, sample_count=0, loss_clamp=1.25)

    assert loss is not None
    assert torch.allclose(loss, torch.tensor(1.25))


def test_robust_depth_mask_keeps_small_sets_unmodified_except_invalid_values():
    depths = np.array([1.0, 2.0, np.inf, -1.0], dtype=np.float32)

    mask = _robust_depth_mask(depths)

    assert mask.tolist() == [True, True, False, False]


def test_should_reset_opacity_skips_final_step():
    assert _should_reset_opacity(1000, 3000, 1000, 0)
    assert _should_reset_opacity(2000, 3000, 1000, 0)
    assert not _should_reset_opacity(3000, 3000, 1000, 0)
    assert not _should_reset_opacity(2500, 3000, 1000, 0)
    assert not _should_reset_opacity(2000, 3000, 0, 0)
    assert not _should_reset_opacity(3000, 4000, 1000, 2000)


def test_reset_opacities_delegates_to_gsplat_reset(monkeypatch):
    captured = {}

    def fake_reset_opa(params, optimizers, state, value):
        captured["params"] = params
        captured["optimizers"] = optimizers
        captured["state"] = state
        captured["value"] = value

    fake_gsplat = types.ModuleType("gsplat")
    fake_strategy = types.ModuleType("gsplat.strategy")
    fake_ops = types.ModuleType("gsplat.strategy.ops")
    fake_ops.reset_opa = fake_reset_opa
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.strategy", fake_strategy)
    monkeypatch.setitem(sys.modules, "gsplat.strategy.ops", fake_ops)

    params = {"opacities": torch.nn.Parameter(torch.zeros(2))}
    optimizers = {"opacities": object()}
    state = {"count": torch.zeros(2)}

    _reset_opacities(params, optimizers, state, 0.02)

    assert captured == {
        "params": params,
        "optimizers": optimizers,
        "state": state,
        "value": 0.02,
    }


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


def test_visible_gaussians_from_info_handles_packed_and_dense_radii():
    device = torch.device("cpu")

    dense = _visible_gaussians_from_info({"radii": torch.tensor([[1.0, 0.0, 2.0]])}, 3, device)
    assert dense is not None
    assert dense.tolist() == [True, False, True]

    dense_xy = _visible_gaussians_from_info(
        {"radii": torch.tensor([[[1.0, 1.0], [1.0, 0.0], [2.0, 2.0]]])},
        3,
        device,
    )
    assert dense_xy is not None
    assert dense_xy.tolist() == [True, False, True]

    packed = _visible_gaussians_from_info({"gaussian_ids": torch.tensor([2, 0, 2])}, 3, device)
    assert packed is not None
    assert packed.tolist() == [True, False, True]


def test_prune_unstable_gaussians_removes_points_outside_scene_radius(monkeypatch):
    captured = {}

    def fake_remove(params, optimizers, strategy_state, mask):
        captured["mask"] = mask.clone()

    fake_gsplat = types.ModuleType("gsplat")
    fake_strategy = types.ModuleType("gsplat.strategy")
    fake_ops = types.ModuleType("gsplat.strategy.ops")
    fake_ops.remove = fake_remove
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.strategy", fake_strategy)
    monkeypatch.setitem(sys.modules, "gsplat.strategy.ops", fake_ops)

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
            ),
            "opacities": torch.nn.Parameter(torch.zeros(3)),
        }
    )
    config = TrainConfig(
        Path("."),
        Path("."),
        Path("."),
        Path("."),
        visibility_min_views=0,
        prune_scene_radius=1.0,
    )

    removed = _prune_unstable_gaussians(params, {}, {}, [], [], config, scene_scale=1.0)

    assert removed == 1
    assert captured["mask"].tolist() == [False, False, True]


def test_train_loop_runs_strategy_post_backward_after_optimizer_step():
    source = __import__("inspect").getsource(_train_gaussians_worker)

    assert source.index("optimizer.step()") < source.index("strategy.step_post_backward")
