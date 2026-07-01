#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vggt_gaussian_reconstruction.frame_selection import select_frames, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames for VGGT reconstruction.")
    parser.add_argument("--video", required=True, type=Path, help="Input scene video.")
    parser.add_argument("--out", required=True, type=Path, help="Output scene directory.")
    parser.add_argument("--num_frames", type=int, default=48)
    parser.add_argument("--strategy", choices=["uniform", "quality"], default="quality")
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    args = parser.parse_args()

    image_dir = args.out / "images"
    metadata = select_frames(
        video=args.video,
        image_dir=image_dir,
        num_frames=args.num_frames,
        strategy=args.strategy,
        candidate_multiplier=args.candidate_multiplier,
    )
    write_metadata(metadata, args.out / "metadata.json")
    print(f"Wrote {metadata['num_frames']} frames to {image_dir}")


if __name__ == "__main__":
    main()
