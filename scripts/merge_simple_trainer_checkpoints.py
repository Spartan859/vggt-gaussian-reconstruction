#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


RANK_RE = re.compile(r"ckpt_(?P<step>\d+)_rank(?P<rank>\d+)\.pt$")
SH_C0 = 0.28209479177387814


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge gsplat simple_trainer rank checkpoints into one viewer checkpoint.")
    parser.add_argument("--scene", type=Path, default=Path("outputs/scene"), help="Scene directory.")
    parser.add_argument("--run-dir", type=Path, default=None, help="simple_trainer run directory containing ckpts/.")
    parser.add_argument("--ckpt-dir", type=Path, default=None, help="Directory containing ckpt_<step>_rank*.pt files.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to merge. Defaults to newest complete step.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output checkpoint path. Defaults to <run-dir>/gaussians_ba/checkpoint.pt when --run-dir is set, else <scene>/gaussians_ba/checkpoint.pt.",
    )
    args = parser.parse_args()

    ckpt_dir = resolve_ckpt_dir(args.scene, args.run_dir, args.ckpt_dir)
    by_step = collect_rank_checkpoints(ckpt_dir)
    if not by_step:
        raise SystemExit(f"No sharded checkpoints found in {ckpt_dir}")

    step = args.step if args.step is not None else max(by_step)
    if step not in by_step:
        available = ", ".join(str(s) for s in sorted(by_step))
        raise SystemExit(f"Step {step} not found in {ckpt_dir}. Available steps: {available}")

    rank_paths = sorted(by_step[step], key=lambda item: item[0])
    out = args.out or default_output_path(args.scene, args.run_dir)
    merge_checkpoints(step, rank_paths, out)


def resolve_ckpt_dir(scene: Path, run_dir: Path | None, ckpt_dir: Path | None) -> Path:
    if ckpt_dir is not None:
        return ckpt_dir
    if run_dir is not None:
        return run_dir / "ckpts"

    runs_dir = scene / "runs"
    runs = [path for path in runs_dir.glob("simple_trainer_ba_*") if (path / "ckpts").is_dir()]
    if not runs:
        raise SystemExit(f"No simple_trainer_ba_* runs with ckpts/ found under {runs_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime) / "ckpts"


def collect_rank_checkpoints(ckpt_dir: Path) -> dict[int, list[tuple[int, Path]]]:
    by_step: dict[int, list[tuple[int, Path]]] = {}
    for path in ckpt_dir.glob("ckpt_*_rank*.pt"):
        match = RANK_RE.match(path.name)
        if not match:
            continue
        step = int(match.group("step"))
        rank = int(match.group("rank"))
        by_step.setdefault(step, []).append((rank, path))
    return by_step


def default_output_path(scene: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir / "gaussians_ba" / "checkpoint.pt"
    return scene / "gaussians_ba" / "checkpoint.pt"


def merge_checkpoints(step: int, rank_paths: list[tuple[int, Path]], out: Path) -> None:
    merged_lists: dict[str, list[torch.Tensor]] = {}
    total = 0
    ranks = [rank for rank, _ in rank_paths]
    expected = list(range(len(ranks)))
    if ranks != expected:
        raise SystemExit(f"Expected contiguous ranks {expected}, found {ranks}")

    print(f"Merging step {step} from {len(rank_paths)} rank checkpoint(s):")
    for rank, path in rank_paths:
        data = torch.load(path, map_location="cpu")
        file_step = int(data.get("step", step))
        if file_step != step:
            raise SystemExit(f"{path} stores step {file_step}, expected {step}")
        splats = data.get("splats")
        if not isinstance(splats, dict):
            raise SystemExit(f"{path} has no splats state dict")
        count = int(splats["means"].shape[0])
        total += count
        print(f"  rank {rank}: {count} Gaussians from {path}")
        for key, value in splats.items():
            merged_lists.setdefault(key, []).append(value.detach().cpu().contiguous())

    keys = sorted(merged_lists)
    first_keys = sorted(merged_lists.keys())
    if keys != first_keys:
        raise SystemExit("Internal key mismatch while merging checkpoints")

    merged = {key: torch.cat(values, dim=0).contiguous() for key, values in merged_lists.items()}
    required = {"means", "opacities", "quats", "scales", "sh0"}
    missing = required - set(merged)
    if missing:
        raise SystemExit(f"Merged checkpoint missing required splat tensors: {sorted(missing)}")

    output = {
        "step": step,
        "splats": merged,
        "means": merged["means"],
        "quats": merged["quats"],
        "scales": merged["scales"].exp(),
        "opacities": merged["opacities"].sigmoid(),
        "sh0": merged["sh0"],
        "colors": (merged["sh0"][:, 0, :] * SH_C0 + 0.5).clamp(0.0, 1.0),
        "source": {
            "format": "gsplat_simple_trainer_sharded",
            "step": step,
            "ranks": ranks,
            "ckpt_dir": str(rank_paths[0][1].parent),
        },
    }
    if "shN" in merged:
        output["shN"] = merged["shN"]

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(output, tmp)
    tmp.replace(out)
    print(f"Wrote {total} Gaussians to {out}")


if __name__ == "__main__":
    main()
