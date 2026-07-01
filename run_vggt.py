#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from vggt_gaussian_reconstruction.vggt_runner import VggtConfig, run_vggt_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT and export COLMAP sparse reconstruction.")
    parser.add_argument("--scene", required=True, type=Path, help="Scene directory with images/.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vggt-repo", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--import-colmap", type=Path, default=None, help="Copy an existing COLMAP sparse model instead of running VGGT.")
    parser.add_argument("--use-ba", action="store_true", help="Deprecated; use this repository's ba_optimize.py stage instead.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf-threshold", type=float, default=5.0)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[], help="Extra args forwarded to VGGT.")
    args = parser.parse_args()

    scene = args.scene
    image_dir = scene / "images"
    out_sparse = scene / "vggt" / "sparse" / "0"
    if not image_dir.exists():
        raise SystemExit(f"Missing images directory: {image_dir}")

    if args.import_colmap:
        if out_sparse.exists():
            shutil.rmtree(out_sparse)
        shutil.copytree(args.import_colmap, out_sparse)
        print(f"Imported COLMAP model to {out_sparse}")
        return

    if args.vggt_repo is not None:
        print("--vggt-repo is deprecated and ignored; using installed vggt package.")
    if args.use_ba:
        print("--use-ba is deprecated here; run ba_optimize.py after VGGT instead.")

    found = run_vggt_package(
        VggtConfig(
            scene=scene,
            device=args.device,
            seed=args.seed,
            conf_threshold=args.conf_threshold,
            max_points=args.max_points,
            extra_args=args.extra_args,
        )
    )
    if found != out_sparse:
        raise SystemExit(f"Unexpected VGGT output path: {found}")
    print(f"VGGT sparse model ready at {out_sparse}")


if __name__ == "__main__":
    main()
