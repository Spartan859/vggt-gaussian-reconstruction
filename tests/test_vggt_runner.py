from __future__ import annotations

from pathlib import Path

import numpy as np

from vggt_gaussian_reconstruction.vggt_runner import _select_ba_points, _write_pycolmap_reconstruction


class DummyReconstruction:
    def __init__(self) -> None:
        self.written_to: str | None = None

    def write(self, path: str) -> None:
        output = Path(path)
        if not output.is_dir():
            raise ValueError(f"missing output directory: {output}")
        self.written_to = path
        (output / "cameras.bin").write_bytes(b"dummy")


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
