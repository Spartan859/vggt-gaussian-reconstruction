#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vggt_gaussian_reconstruction.eval_utils import image_metrics, reconstruction_summary, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reconstruction and optional render metrics.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--render-dir-vggt", type=Path, default=None)
    parser.add_argument("--render-dir-ba", type=Path, default=None)
    args = parser.parse_args()

    report = {}
    for mode in ["vggt", "ba"]:
        sparse = args.scene / mode / "sparse" / "0"
        if sparse.exists():
            report[mode] = {"reconstruction": reconstruction_summary(sparse)}

    if args.render_dir_vggt:
        report.setdefault("vggt", {})["render_metrics"] = image_metrics(args.render_dir_vggt, args.scene / "images")
    if args.render_dir_ba:
        report.setdefault("ba", {})["render_metrics"] = image_metrics(args.render_dir_ba, args.scene / "images")

    out = args.scene / "eval_report.json"
    write_json(report, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
