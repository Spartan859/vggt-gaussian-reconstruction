from __future__ import annotations

from pathlib import Path
import os

import torch

from vggt_gaussian_reconstruction.colmap import Camera, Image, Reconstruction, write_model
from vggt_gaussian_reconstruction.geometry import rotmat_to_qvec
from viewer import (
    _checkpoint_choices,
    _filter_splats,
    _scene_for_checkpoint,
    generate_camera_path,
    load_scene_camera_poses,
    look_at_c2w,
    discover_checkpoints,
    discover_viewer_scenes,
    load_splats,
    resolve_checkpoint,
)


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "splats": {
                "means": torch.zeros((2, 3)),
                "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
                "scales": torch.zeros((2, 3)),
                "opacities": torch.zeros((2,)),
                "sh0": torch.zeros((2, 1, 3)),
                "shN": torch.zeros((2, 15, 3)),
            }
        },
        path,
    )


def test_resolve_checkpoint_prefers_newest_run(tmp_path: Path):
    scene = tmp_path / "scene"
    older = scene / "runs" / "old" / "gaussians_ba" / "checkpoint.pt"
    newer = scene / "runs" / "new" / "gaussians_ba" / "checkpoint.pt"
    _write_checkpoint(older)
    _write_checkpoint(newer)

    os.utime(older, (10, 10))
    os.utime(newer, (20, 20))

    assert resolve_checkpoint(scene, "ba") == newer


def test_discover_checkpoints_returns_newest_first(tmp_path: Path):
    scene = tmp_path / "scene"
    direct = scene / "gaussians_ba" / "checkpoint.pt"
    older = scene / "runs" / "old" / "gaussians_ba" / "checkpoint.pt"
    newer = scene / "runs" / "new" / "gaussians_ba" / "checkpoint.pt"
    for path in (direct, older, newer):
        _write_checkpoint(path)
    os.utime(direct, (5, 5))
    os.utime(older, (10, 10))
    os.utime(newer, (20, 20))

    assert discover_checkpoints(scene, "ba") == [newer, older, direct]


def test_discover_viewer_scenes_includes_default_sibling_datasets(tmp_path: Path):
    scene = tmp_path / "outputs" / "scene"
    data1 = tmp_path / "outputs" / "data1_baseline"
    data2 = tmp_path / "outputs" / "data2_baseline"
    unrelated = tmp_path / "outputs" / "other"
    for path in (scene, data1, data2, unrelated):
        _write_checkpoint(path / "runs" / "run" / "gaussians_ba" / "checkpoint.pt")

    assert discover_viewer_scenes(scene, "ba") == [scene, data1, data2]


def test_scene_for_checkpoint_maps_run_checkpoint_to_dataset_root(tmp_path: Path):
    scene = tmp_path / "outputs" / "scene"
    data1 = tmp_path / "outputs" / "data1_baseline"
    checkpoint = data1 / "runs" / "data1_maskaware" / "gaussians_ba" / "checkpoint.pt"
    _write_checkpoint(checkpoint)

    assert _scene_for_checkpoint(checkpoint, [scene, data1]) == data1


def test_checkpoint_choices_disambiguates_duplicate_labels(tmp_path: Path):
    first = tmp_path / "a" / "gaussians_ba" / "checkpoint.pt"
    second = tmp_path / "b" / "gaussians_ba" / "checkpoint.pt"

    labels, by_label = _checkpoint_choices([first, second])

    assert labels == ["a/gaussians_ba", "b/gaussians_ba"]
    assert by_label[labels[0]] == first
    assert by_label[labels[1]] == second


def test_load_splats_reads_repository_checkpoint_format(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)

    splats, sh_degree = load_splats(checkpoint, torch.device("cpu"))

    assert sh_degree == 3
    assert splats["means"].shape == (2, 3)
    assert splats["colors"].shape == (2, 16, 3)
    assert splats["colors"].is_contiguous()


def test_filter_splats_removes_low_opacity_large_scale_and_far_points():
    splats = {
        "means": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "quats": torch.ones((4, 4), dtype=torch.float32),
        "scales": torch.log(torch.tensor([[0.02, 0.02, 0.02], [0.2, 0.2, 0.2], [0.02, 0.02, 0.02], [0.02, 0.02, 0.02]])),
        "opacities": torch.logit(torch.tensor([0.8, 0.8, 0.8, 0.01])),
        "colors": torch.zeros((4, 16, 3), dtype=torch.float32),
    }

    filtered = _filter_splats(splats, min_opacity=0.05, max_scale=0.08, max_radius=2.8, max_gaussians=0)

    assert filtered["means"].shape == (1, 3)
    assert torch.allclose(filtered["means"][0], torch.zeros(3))


def test_filter_splats_caps_by_opacity_per_scale_score():
    splats = {
        "means": torch.zeros((3, 3), dtype=torch.float32),
        "quats": torch.ones((3, 4), dtype=torch.float32),
        "scales": torch.log(torch.tensor([[0.01, 0.01, 0.01], [0.10, 0.10, 0.10], [0.02, 0.02, 0.02]])),
        "opacities": torch.logit(torch.tensor([0.3, 0.9, 0.6])),
        "colors": torch.zeros((3, 16, 3), dtype=torch.float32),
    }

    filtered = _filter_splats(splats, max_gaussians=2)

    assert filtered["means"].shape == (2, 3)
    assert torch.allclose(filtered["opacities"].sigmoid().sort().values, torch.tensor([0.3, 0.6]))


def test_look_at_c2w_uses_opencv_camera_convention():
    c2w = look_at_c2w(
        position=torch.tensor([0.0, 0.0, -1.0]).numpy(),
        target=torch.tensor([0.0, 0.0, 0.0]).numpy(),
        up=torch.tensor([0.0, 1.0, 0.0]).numpy(),
    )

    assert c2w.shape == (4, 4)
    assert torch.allclose(torch.from_numpy(c2w[:3, 2]).float(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    assert torch.allclose(torch.from_numpy(-c2w[:3, 1]).float(), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(torch.from_numpy(c2w[:3, :3].T @ c2w[:3, :3]).float(), torch.eye(3), atol=1e-6)


def test_generate_camera_path_interpolates_requested_frame_count():
    poses = []
    for x in (0.0, 1.0, 2.0):
        c2w = torch.eye(4).numpy()
        c2w[0, 3] = x
        poses.append({"c2w": c2w, "fov": 1.0})

    path = generate_camera_path(poses, "interp", 7)

    assert len(path) == 7
    assert path[0]["c2w"][0, 3] == 0.0
    assert path[-1]["c2w"][0, 3] == 2.0


def test_generate_camera_path_orbit_faces_scene_center():
    poses = []
    for x in (-1.0, 1.0):
        c2w = torch.eye(4).numpy()
        c2w[0, 3] = x
        poses.append({"c2w": c2w, "fov": 1.0})

    path = generate_camera_path(poses, "ellipse", 8)

    assert len(path) == 8
    for pose in path:
        c2w = pose["c2w"]
        position = c2w[:3, 3]
        forward = c2w[:3, 2]
        to_center = -position
        if torch.linalg.norm(torch.from_numpy(to_center).float()) > 1e-6:
            to_center = to_center / (to_center**2).sum() ** 0.5
            assert float(forward @ to_center) > 0.99


def test_load_scene_camera_poses_reads_normalized_sparse_model(tmp_path: Path):
    scene = tmp_path / "scene"
    sparse = scene / "ba" / "sparse" / "0"
    camera = Camera(1, "PINHOLE", 8, 6, torch.tensor([4.0, 4.0, 4.0, 3.0]).numpy())
    image = Image(1, rotmat_to_qvec(torch.eye(3).numpy()), torch.tensor([0.0, 0.0, 1.0]).numpy(), 1, "im1.png")
    write_model(Reconstruction({1: camera}, {1: image}, {}), sparse)

    poses = load_scene_camera_poses(scene, "ba")

    assert len(poses) == 1
    assert poses[0]["c2w"].shape == (4, 4)
    assert poses[0]["fov"] > 0
