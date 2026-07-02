#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch


SH_C0 = 0.28209479177387814


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove likely floater Gaussians from a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input checkpoint.pt.")
    parser.add_argument("--out", type=Path, required=True, help="Output checkpoint.pt.")
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=0.08)
    parser.add_argument("--max-radius", type=float, default=2.8)
    parser.add_argument("--max-gaussians", type=int, default=120_000)
    args = parser.parse_args()

    data = torch.load(args.checkpoint, map_location="cpu")
    splats = data.get("splats")
    if not isinstance(splats, dict):
        raise SystemExit(f"{args.checkpoint} has no splats state dict")

    keep = build_keep_mask(
        splats,
        min_opacity=args.min_opacity,
        max_scale=args.max_scale,
        max_radius=args.max_radius,
        max_gaussians=args.max_gaussians,
    )
    if not bool(keep.any()):
        raise SystemExit("Cleaning thresholds removed every Gaussian; relax thresholds.")

    cleaned = {}
    for key, value in splats.items():
        if torch.is_tensor(value) and value.shape[:1] == keep.shape:
            cleaned[key] = value[keep].contiguous()
        else:
            cleaned[key] = value

    output = {
        "step": data.get("step", 0),
        "splats": cleaned,
        "means": cleaned["means"],
        "quats": cleaned["quats"],
        "scales": cleaned["scales"].exp(),
        "opacities": cleaned["opacities"].sigmoid(),
        "sh0": cleaned["sh0"],
        "colors": (cleaned["sh0"][:, 0, :] * SH_C0 + 0.5).clamp(0.0, 1.0),
        "source": {
            "format": "cleaned_gaussian_checkpoint",
            "input": str(args.checkpoint),
            "input_count": int(keep.numel()),
            "output_count": int(keep.sum().item()),
            "min_opacity": args.min_opacity,
            "max_scale": args.max_scale,
            "max_radius": args.max_radius,
            "max_gaussians": args.max_gaussians,
        },
    }
    if "shN" in cleaned:
        output["shN"] = cleaned["shN"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(output, tmp)
    tmp.replace(args.out)
    print(f"Kept {int(keep.sum().item())} / {int(keep.numel())} Gaussians")
    print(f"Wrote {args.out}")


def build_keep_mask(
    splats: dict[str, torch.Tensor],
    min_opacity: float,
    max_scale: float,
    max_radius: float,
    max_gaussians: int,
) -> torch.Tensor:
    means = splats["means"]
    opacities = splats["opacities"].sigmoid()
    scales = splats["scales"].exp().max(dim=-1).values
    radius = torch.linalg.norm(means, dim=-1)

    keep = torch.isfinite(means).all(dim=-1)
    keep &= torch.isfinite(opacities) & torch.isfinite(scales) & torch.isfinite(radius)
    if min_opacity > 0:
        keep &= opacities >= min_opacity
    if max_scale > 0:
        keep &= scales <= max_scale
    if max_radius > 0:
        keep &= radius <= max_radius

    if max_gaussians > 0 and int(keep.sum().item()) > max_gaussians:
        score = opacities / scales.clamp_min(1e-6)
        score[~keep] = -torch.inf
        ids = torch.topk(score, k=max_gaussians, largest=True).indices
        capped = torch.zeros_like(keep)
        capped[ids] = True
        keep = capped
    return keep


if __name__ == "__main__":
    main()
