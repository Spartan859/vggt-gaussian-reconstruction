#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from vggt_gaussian_reconstruction.colmap import has_model
from vggt_gaussian_reconstruction.gaussian_trainer import TrainConfig, train_gaussians


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D Gaussians from VGGT or BA COLMAP output.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--mode", choices=["vggt", "ba"], default="ba")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1.0, help="Global multiplier for gsplat-style parameter learning rates.")
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--max-gaussians", type=int, default=200_000)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--refine-stop-iter", type=int, default=15_000)
    parser.add_argument("--grow-grad2d", type=float, default=0.0002)
    parser.add_argument("--prune-opa", type=float, default=0.01)
    parser.add_argument("--min-track-len", type=int, default=4)
    parser.add_argument("--opacity-reg", type=float, default=1e-3)
    parser.add_argument("--scale-reg", type=float, default=1e-3)
    parser.add_argument("--prune-every", type=int, default=1000)
    parser.add_argument("--prune-large-scale", type=float, default=0.25)
    parser.add_argument("--visibility-prune-every", type=int, default=2000)
    parser.add_argument("--visibility-prune-start", type=int, default=8000)
    parser.add_argument("--visibility-min-views", type=int, default=2)
    parser.add_argument("--prune-scene-radius", type=float, default=2.5)
    parser.add_argument("--depth-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-lambda", type=float, default=1e-2)
    parser.add_argument("--depth-sample-count", type=int, default=2048)
    parser.add_argument("--depth-loss-clamp", type=float, default=2.0)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Optional foreground masks directory. Defaults to SCENE_DIR/masks when it exists.",
    )
    parser.add_argument("--mask-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-alpha-lambda", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--opacities-lr", type=float, default=2e-2)
    parser.add_argument("--opacity-reset-every", type=int, default=1000)
    parser.add_argument("--opacity-reset-until", type=int, default=0)
    parser.add_argument("--final-prune-opa", type=float, default=0.05)
    parser.add_argument("--final-prune-large-scale", type=float, default=0.08)
    parser.add_argument("--final-prune-scene-radius", type=float, default=2.8)
    parser.add_argument("--final-visibility-min-views", type=int, default=3)
    parser.add_argument("--distributed", action="store_true", help="Shard Gaussians across all visible CUDA devices.")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--render-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    sparse = args.scene / args.mode / "sparse" / "0"
    images = args.scene / "images"
    mask_dir = args.mask_dir if args.mask_dir is not None else args.scene / "masks"
    if not mask_dir.exists():
        mask_dir = None
    output = args.output_dir if args.output_dir is not None else args.scene / f"gaussians_{args.mode}"
    if not has_model(sparse):
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
            max_gaussians=args.max_gaussians,
            opacities_lr=args.opacities_lr,
            test_every=args.test_every,
            refine_stop_iter=args.refine_stop_iter,
            grow_grad2d=args.grow_grad2d,
            prune_opa=args.prune_opa,
            min_track_len=args.min_track_len,
            opacity_reg=args.opacity_reg,
            scale_reg=args.scale_reg,
            prune_every=args.prune_every,
            prune_large_scale=args.prune_large_scale,
            visibility_prune_every=args.visibility_prune_every,
            visibility_prune_start=args.visibility_prune_start,
            visibility_min_views=args.visibility_min_views,
            prune_scene_radius=args.prune_scene_radius,
            depth_loss=args.depth_loss,
            depth_lambda=args.depth_lambda,
            depth_sample_count=args.depth_sample_count,
            depth_loss_clamp=args.depth_loss_clamp,
            mask_dir=mask_dir,
            mask_loss=args.mask_loss,
            mask_alpha_lambda=args.mask_alpha_lambda,
            mask_threshold=args.mask_threshold,
            opacity_reset_every=args.opacity_reset_every,
            opacity_reset_until=args.opacity_reset_until,
            final_prune_opa=args.final_prune_opa,
            final_prune_large_scale=args.final_prune_large_scale,
            final_prune_scene_radius=args.final_prune_scene_radius,
            final_visibility_min_views=args.final_visibility_min_views,
            distributed=args.distributed,
            save_every=args.save_every,
            render_every=args.render_every,
        )
    )


if __name__ == "__main__":
    main()
