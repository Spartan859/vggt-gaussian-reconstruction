#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pycolmap

from vggt_gaussian_reconstruction.eval_utils import reconstruction_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine VGGT reconstruction with pycolmap bundle adjustment.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--input-sparse", type=Path, default=None)
    parser.add_argument("--output-sparse", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=1000, help=argparse.SUPPRESS)
    parser.add_argument("--lr_pose", type=float, default=1e-3, help=argparse.SUPPRESS)
    parser.add_argument("--lr_points", type=float, default=1e-2, help=argparse.SUPPRESS)
    parser.add_argument("--huber_delta", type=float, default=4.0, help=argparse.SUPPRESS)
    parser.add_argument("--min_track_len", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    args = parser.parse_args()

    input_sparse = args.input_sparse or args.scene / "vggt" / "sparse" / "0"
    output_sparse = args.output_sparse or args.scene / "ba" / "sparse" / "0"
    stats = run_pycolmap_ba(input_sparse, output_sparse)
    stats_path = args.scene / "ba" / "ba_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    before = stats["before"]
    after = stats["after"]
    print(
        "pycolmap BA points/observations: "
        f"{before['points3d']}/{before['observations']} -> {after['points3d']}/{after['observations']}"
    )
    print(f"Wrote refined model to {output_sparse}")


def run_pycolmap_ba(input_sparse: Path, output_sparse: Path) -> dict:
    before = reconstruction_summary(input_sparse)
    reconstruction = pycolmap.Reconstruction(str(input_sparse))
    pycolmap.bundle_adjustment(reconstruction, pycolmap.BundleAdjustmentOptions())
    if output_sparse.exists():
        shutil.rmtree(output_sparse)
    output_sparse.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(output_sparse))
    after = reconstruction_summary(output_sparse)
    return {
        "backend": "pycolmap",
        "input_sparse": str(input_sparse),
        "output_sparse": str(output_sparse),
        "before": before,
        "after": after,
    }


if __name__ == "__main__":
    main()
