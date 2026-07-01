from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_ba_optimize():
    path = Path(__file__).resolve().parents[1] / "ba_optimize.py"
    spec = importlib.util.spec_from_file_location("ba_optimize", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyReconstruction:
    def __init__(self, path: str) -> None:
        self.path = path

    def write(self, path: str) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "cameras.bin").write_bytes(b"camera")


def test_run_pycolmap_ba_writes_output_and_stats(monkeypatch, tmp_path: Path) -> None:
    module = _load_ba_optimize()
    input_sparse = tmp_path / "vggt" / "sparse" / "0"
    output_sparse = tmp_path / "ba" / "sparse" / "0"
    input_sparse.mkdir(parents=True)
    calls = []

    monkeypatch.setattr(
        module,
        "reconstruction_summary",
        lambda path: {"points3d": 1 if path == input_sparse else 2, "observations": 3 if path == input_sparse else 4},
    )
    monkeypatch.setattr(module.pycolmap, "Reconstruction", DummyReconstruction)
    monkeypatch.setattr(module.pycolmap, "BundleAdjustmentOptions", lambda: object())
    monkeypatch.setattr(module.pycolmap, "bundle_adjustment", lambda reconstruction, options: calls.append(reconstruction))

    stats = module.run_pycolmap_ba(input_sparse, output_sparse)

    assert calls and calls[0].path == str(input_sparse)
    assert (output_sparse / "cameras.bin").exists()
    assert stats["backend"] == "pycolmap"
    assert stats["before"]["observations"] == 3
    assert stats["after"]["observations"] == 4
