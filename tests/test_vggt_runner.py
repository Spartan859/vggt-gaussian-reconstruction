from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vggt_gaussian_reconstruction.vggt_runner import (
    _configure_torch_hub,
    _configure_vggsfm_tracker_loader,
    _prepare_tracker_inputs,
    _rename_and_rescale,
    _select_ba_points,
    _write_pycolmap_reconstruction,
)


class DummyReconstruction:
    def __init__(self) -> None:
        self.written_to: str | None = None

    def write(self, path: str) -> None:
        output = Path(path)
        if not output.is_dir():
            raise ValueError(f"missing output directory: {output}")
        self.written_to = path
        (output / "cameras.bin").write_bytes(b"dummy")


class DummyPoint2D:
    def __init__(self, xy):
        self.xy = np.array(xy, dtype=np.float64)


class DummyCamera:
    def __init__(self) -> None:
        self.params = np.array([100.0, 100.0, 50.0, 50.0], dtype=np.float64)
        self.width = 100
        self.height = 100


class DummyImage:
    def __init__(self) -> None:
        self.camera_id = 1
        self.name = "old.png"
        self.points2D = [DummyPoint2D([20.0, 30.0])]


class DummyPycolmapReconstruction:
    def __init__(self) -> None:
        self.cameras = {1: DummyCamera()}
        self.images = {1: DummyImage()}


def test_write_pycolmap_reconstruction_creates_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "scene" / "vggt" / "sparse" / "0"
    old_file = output_dir / "old.bin"
    output_dir.mkdir(parents=True)
    old_file.write_bytes(b"old")

    reconstruction = DummyReconstruction()
    _write_pycolmap_reconstruction(reconstruction, output_dir)

    assert reconstruction.written_to == str(output_dir)
    assert (output_dir / "cameras.bin").exists()
    assert not old_file.exists()


def test_rename_and_rescale_shifts_ba_tracks_to_original_resolution() -> None:
    reconstruction = DummyPycolmapReconstruction()
    original_coords = np.array([[10.0, 20.0, 200.0, 100.0]], dtype=np.float64)

    _rename_and_rescale(
        reconstruction,
        ["frame.png"],
        original_coords,
        image_size=100,
        shift_point2d_to_original_res=True,
    )

    camera = reconstruction.cameras[1]
    image = reconstruction.images[1]
    assert image.name == "frame.png"
    assert camera.width == 200
    assert camera.height == 100
    assert np.allclose(camera.params, [200.0, 200.0, 100.0, 50.0])
    assert np.allclose(image.points2D[0].xy, [20.0, 20.0])


def test_select_ba_points_falls_back_to_finite_points() -> None:
    world_points = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]],
        dtype=np.float32,
    )
    world_points_conf = np.array([[[0.1, 0.2]]], dtype=np.float32)
    points_rgb = np.array([[[[10, 20, 30], [40, 50, 60]]]], dtype=np.uint8)

    points_3d, points_rgb_out, stats = _select_ba_points(
        world_points,
        world_points_conf,
        points_rgb,
        max_points=10,
        conf_threshold=10.0,
    )

    assert points_3d.shape == (2, 3)
    assert points_rgb_out.shape == (2, 3)
    assert stats["selected"] == 2


def test_select_ba_points_accepts_channel_first_rgb() -> None:
    world_points = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]],
        dtype=np.float32,
    )
    world_points_conf = np.array([[[0.9, 0.9]]], dtype=np.float32)
    points_rgb = np.array(
        [[
            [[10, 20]],
            [[30, 40]],
            [[50, 60]],
        ]],
        dtype=np.uint8,
    )

    points_3d, points_rgb_out, stats = _select_ba_points(
        world_points,
        world_points_conf,
        points_rgb,
        max_points=10,
        conf_threshold=0.5,
    )

    assert points_3d.shape == (2, 3)
    assert points_rgb_out.shape == (2, 3)
    assert stats["selected"] == 2


def test_configure_torch_hub_uses_torch_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))

    _configure_torch_hub()

    assert Path(torch.hub.get_dir()) == tmp_path / "torch" / "hub"


def test_configure_vggsfm_tracker_loader_replaces_dino_ranking() -> None:
    from vggt.dependency import track_predict

    _configure_vggsfm_tracker_loader()
    ranks = track_predict.generate_rank_by_dino(np.zeros((5, 3, 4, 4)), query_frame_num=3, device="cpu")

    assert ranks == [0, 2, 4]


def test_prepare_tracker_inputs_keeps_track_filters_on_cpu() -> None:
    depth_conf = torch.ones((1, 2, 3), dtype=torch.float16)
    points_3d = np.ones((2, 3, 3), dtype=np.float64)

    conf, points = _prepare_tracker_inputs(depth_conf, points_3d)

    assert conf.device.type == "cpu"
    assert points.device.type == "cpu"
    assert conf.dtype == torch.float32
    assert points.dtype == torch.float32
    assert conf.shape == (2, 3)
    assert points.shape == (2, 3, 3)
