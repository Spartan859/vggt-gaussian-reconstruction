#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from vggt_gaussian_reconstruction.gaussian_trainer import TrainConfig, train_gaussians


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D Gaussians from VGGT or BA COLMAP output.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--mode", choices=["vggt", "ba"], default="ba")
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--render-every", type=int, default=1000)
    args = parser.parse_args()

    sparse = args.scene / args.mode / "sparse" / "0"
    images = args.scene / "images"
    output = args.scene / f"gaussians_{args.mode}"
    if not (sparse / "cameras.txt").exists():
        raise SystemExit(f"Missing sparse model: {sparse}")
    if not images.exists():
        raise SystemExit(f"Missing images directory: {images}")

    train_gaussians(
        TrainConfig(
            scene=args.scene,
            sparse_dir=sparse,
            image_dir=images,
            output_dir=output,
            steps=args.steps,
            device=args.device,
            lr=args.lr,
            image_scale=args.image_scale,
            max_points=args.max_points,
            save_every=args.save_every,
            render_every=args.render_every,
        )
    )


if __name__ == "__main__":
    main()
