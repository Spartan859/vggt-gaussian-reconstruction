#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch an interactive Gaussian viewer command.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--mode", choices=["vggt", "ba"], default="ba")
    parser.add_argument("--command-template", default=None)
    args = parser.parse_args()

    model_dir = args.scene / f"gaussians_{args.mode}"
    if not model_dir.exists():
        raise SystemExit(f"Missing Gaussian output directory: {model_dir}")
    if args.command_template is None:
        raise SystemExit(
            "Pass --command-template for the viewer installed on your CUDA machine, "
            "for example: 'python viewer.py --model {model}'"
        )
    command = args.command_template.format(model=model_dir, scene=args.scene, mode=args.mode)
    subprocess.run(command.split(), check=True)


if __name__ == "__main__":
    main()
