# VGGT Gaussian Reconstruction

This repository contains a complete implementation scaffold for the assignment:
multi-view reconstruction with VGGT initialization, custom Bundle Adjustment,
3D Gaussian optimization, and evaluation for the PPT/demo.

The default target is the scene video in `data/scene.mp4` or any explicit video
path passed to `prepare_data.py`. In this workspace the provided data directory
is ignored by git, so commands below use explicit paths.

## Environment

Local CPU machines can run data preparation, BA unit tests, and smoke checks.
The full VGGT and Gaussian training path requires a CUDA GPU.

```bash
python -m pip install -e ".[dev,metrics]"
```

On the CUDA machine, install VGGT plus the matching `gsplat` wheel. For the
Baidu A800 environment, `scripts/setup_a800_env.sh` installs PyTorch
`2.3.1+cu118` and `gsplat==1.5.3+pt23cu118`.

## End-to-end commands

```bash
python prepare_data.py --video "大作业数据/数据3-场景.mp4" --out outputs/scene --num_frames 96
python run_vggt.py --scene outputs/scene --device cuda
python ba_optimize.py --scene outputs/scene --iters 1000 --lr_pose 1e-3 --lr_points 1e-2
python train_gaussians.py --scene outputs/scene --mode vggt --steps 30000
python train_gaussians.py --scene outputs/scene --mode ba --steps 30000
python evaluate.py --scene outputs/scene
python viewer.py --scene outputs/scene --mode ba
```

## Improvement experiment

Run frame extraction twice and compare:

```bash
python prepare_data.py --video "大作业数据/数据3-场景.mp4" --out outputs/scene_uniform --strategy uniform --num_frames 48
python prepare_data.py --video "大作业数据/数据3-场景.mp4" --out outputs/scene_selected --strategy quality --num_frames 96
```

Use the same VGGT, BA, Gaussian, and evaluation commands for both directories.
The PPT can report reprojection error, render metrics, training time, and FPS
for uniform sampling versus quality-aware frame selection.

## Output layout

```text
outputs/scene/
  images/
  metadata.json
  vggt/sparse/0/{cameras.txt,images.txt,points3D.txt}
  ba/sparse/0/{cameras.txt,images.txt,points3D.txt}
  gaussians_vggt/
  gaussians_ba/
  eval_report.json
```

## Notes

- `run_vggt.py` uses the installed `vggt` package, or an already-exported
  COLMAP sparse model supplied with `--import-colmap`.
- `ba_optimize.py` is a local PyTorch implementation. It refines per-image
  SE(3) deltas and 3D points from COLMAP observations using Huber reprojection
  loss.
- `train_gaussians.py` uses this repository's local trainer on top of the
  installed `gsplat` package. It reads `images/` plus the selected COLMAP sparse
  model, optimizes Gaussians, and writes checkpoints/renders under
  `gaussians_vggt/` or `gaussians_ba/`.
- `docs/ppt_outline.md` maps the implementation and experiments to the required
  presentation sections.
