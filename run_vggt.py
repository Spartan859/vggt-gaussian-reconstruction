#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT and export COLMAP sparse reconstruction.")
    parser.add_argument("--scene", required=True, type=Path, help="Scene directory with images/.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vggt-repo", type=Path, default=None, help="Path to a VGGT checkout containing demo_colmap.py.")
    parser.add_argument("--import-colmap", type=Path, default=None, help="Copy an existing COLMAP sparse model instead of running VGGT.")
    parser.add_argument("--use-ba", action="store_true", help="Pass VGGT's own BA flag if supported.")
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

    if args.vggt_repo is None:
        raise SystemExit("Pass --vggt-repo /path/to/vggt or --import-colmap /path/to/sparse/0")

    demo = args.vggt_repo / "demo_colmap.py"
    if not demo.exists():
        raise SystemExit(f"Cannot find VGGT demo_colmap.py at {demo}")

    out_dir = scene / "vggt"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(demo),
        "--scene_dir",
        str(scene),
        "--device",
        args.device,
    ]
    if args.use_ba:
        cmd.append("--use_ba")
    cmd.extend(args.extra_args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    candidates = [
        scene / "sparse" / "0",
        scene / "sparse",
        out_dir / "sparse" / "0",
        out_dir / "sparse",
    ]
    found = next((p for p in candidates if (p / "cameras.txt").exists() and (p / "images.txt").exists()), None)
    if found is None:
        raise SystemExit("VGGT finished, but no COLMAP text sparse model was found. Check VGGT output.")
    if found != out_sparse:
        if out_sparse.exists():
            shutil.rmtree(out_sparse)
        shutil.copytree(found, out_sparse)
    print(f"VGGT sparse model ready at {out_sparse}")


if __name__ == "__main__":
    main()
