# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ reconstruction pipeline. Top-level scripts are the main entrypoints: `prepare_data.py`, `run_vggt.py`, `ba_optimize.py`, `train_gaussians.py`, `evaluate.py`, and `viewer.py`. Reusable code lives under `src/vggt_gaussian_reconstruction/` for COLMAP IO, BA, geometry, frame selection, and evaluation helpers. Tests are in `tests/`. Generated assets belong in `outputs/`; local data such as `大作业数据/` is ignored by git.

## Build, Test, and Development Commands

Install development extras for local checks:

```bash
python -m pip install -e ".[dev,metrics]"
```

Run the test suite:

```bash
pytest
```

On Baidu A800 use the `VGGT_GSPLAT_A800` micromamba environment. Launch the full pipeline with:

```bash
bash scripts/train_infer_baidu.sh
```

Resume after VGGT/BA outputs exist:

```bash
bash scripts/train_infer_baidu.sh --resume-from gsplat
```

Manual pipeline commands:

```bash
python prepare_data.py --video "大作业数据/数据3-场景.mp4" --out outputs/scene --num_frames 96
python run_vggt.py --scene outputs/scene --device cuda
python ba_optimize.py --scene outputs/scene --iters 1000 --lr_pose 1e-3 --lr_points 1e-2
python train_gaussians.py --scene outputs/scene --mode ba --steps 30000
python evaluate.py --scene outputs/scene --output outputs/scene/eval_report.json
python viewer.py --scene outputs/scene --mode ba
```

Full VGGT and Gaussian training require CUDA plus installed `vggt` and `gsplat`. CPU machines can run preparation, tests, and syntax checks.

## Coding Style & Naming Conventions

Use 4-space indentation, explicit imports, and type hints where they clarify data flow. Keep reusable logic in `src/vggt_gaussian_reconstruction/`; keep top-level scripts as thin CLI layers. Use `snake_case` for modules/functions, `PascalCase` for classes, and `test_*.py` with `test_*` functions. Prefer `pathlib.Path` and structured COLMAP helpers. Keep the Gaussian trainer aligned with `external/gsplat/examples/simple_trainer.py`: SH colors, per-parameter Adam optimizers, `DefaultStrategy`, SSIM/L1 loss, and scheduled SH degree.

## Testing Guidelines

Pytest is configured in `pyproject.toml`. Add focused tests for geometry, frame selection, COLMAP conversion, BA, and Gaussian trainer changes. Use `tmp_path` and small synthetic data. Run:

```bash
/mnt/share/micromamba/root/envs/VGGT_GSPLAT_A800/bin/python -m pytest
```

## Commit & Pull Request Guidelines

Use concise imperative commit subjects, for example `Fix COLMAP camera export` or `Align Gaussian training with gsplat simple trainer`. PRs should describe the affected pipeline stage, commands run, and required assets such as CUDA, VGGT, `gsplat`, or local data. Include screenshots or metrics for viewer or quality changes.

## Security & Configuration Tips

Do not commit local datasets, generated reconstructions, checkpoints, or cache files. `.gitignore` excludes `大作业数据/`, `outputs/`, caches, and package metadata. `scripts/train_infer_baidu.sh` preserves each run under `SCENE_DIR/runs/EXP_NAME`; do not overwrite prior outputs unless explicitly requested. Keep machine-specific paths out of source files.
