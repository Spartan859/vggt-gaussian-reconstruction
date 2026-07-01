from __future__ import annotations

from pathlib import Path

from vggt_gaussian_reconstruction.vggt_runner import _write_pycolmap_reconstruction


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
