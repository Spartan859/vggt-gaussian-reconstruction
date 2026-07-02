#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vggt_gaussian_reconstruction.colmap import read_model, write_model
from vggt_gaussian_reconstruction.eval_utils import reconstruction_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter COLMAP points by track length and reprojection error.")
    parser.add_argument("--input-sparse", required=True, type=Path)
    parser.add_argument("--output-sparse", required=True, type=Path)
    parser.add_argument("--min-track-len", type=int, default=3)
    parser.add_argument("--max-error", type=float, default=3.0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    stats = filter_points(args.input_sparse, args.output_sparse, args.min_track_len, args.max_error)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def filter_points(input_sparse: Path, output_sparse: Path, min_track_len: int, max_error: float) -> dict:
    before = reconstruction_summary(input_sparse)
    model = read_model(input_sparse)
    kept_ids = {
        point_id
        for point_id, point in model.points3d.items()
        if len(point.track) >= min_track_len and np.isfinite(point.error) and point.error <= max_error
    }
    model.points3d = {point_id: point for point_id, point in model.points3d.items() if point_id in kept_ids}
    for image in model.images.values():
        image.point3d_ids = np.array([pid if int(pid) in kept_ids else -1 for pid in image.point3d_ids], dtype=np.int64)
    if output_sparse.exists():
        shutil.rmtree(output_sparse)
    write_model(model, output_sparse)
    after = reconstruction_summary(output_sparse)
    return {
        "input_sparse": str(input_sparse),
        "output_sparse": str(output_sparse),
        "min_track_len": min_track_len,
        "max_error": max_error,
        "before": before,
        "after": after,
    }


if __name__ == "__main__":
    main()
