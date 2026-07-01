#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vggt_gaussian_reconstruction.ba import optimize_reconstruction
from vggt_gaussian_reconstruction.colmap import read_model, write_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine VGGT cameras and points with custom PyTorch BA.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--input-sparse", type=Path, default=None)
    parser.add_argument("--output-sparse", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr_pose", type=float, default=1e-3)
    parser.add_argument("--lr_points", type=float, default=1e-2)
    parser.add_argument("--huber_delta", type=float, default=4.0)
    parser.add_argument("--min_track_len", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    input_sparse = args.input_sparse or args.scene / "vggt" / "sparse" / "0"
    output_sparse = args.output_sparse or args.scene / "ba" / "sparse" / "0"
    model = read_model(input_sparse)
    refined, stats = optimize_reconstruction(
        model,
        iters=args.iters,
        lr_pose=args.lr_pose,
        lr_points=args.lr_points,
        huber_delta=args.huber_delta,
        min_track_len=args.min_track_len,
        device=args.device,
    )
    write_model(refined, output_sparse)
    stats_path = args.scene / "ba" / "ba_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")
    print(f"BA RMSE: {stats.initial_rmse:.4f} -> {stats.final_rmse:.4f} px")
    print(f"Wrote refined model to {output_sparse}")


if __name__ == "__main__":
    main()
