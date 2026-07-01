# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ reconstruction pipeline. Top-level scripts are the main entrypoints: `prepare_data.py`, `run_vggt.py`, `ba_optimize.py`, `train_gaussians.py`, `evaluate.py`, and `viewer.py`. Reusable code lives under `src/vggt_gaussian_reconstruction/` for COLMAP IO, bundle adjustment, geometry, frame selection, and evaluation helpers. Tests are in `tests/`. `docs/ppt_outline.md` contains presentation notes. Generated assets belong in `outputs/`; local data such as `大作业数据/` is ignored by git.

## Build, Test, and Development Commands

Install the package and development extras:

```bash
python -m pip install -e ".[dev,metrics]"
```

Run the test suite:

```bash
pytest
```

Typical local pipeline:

```bash
python prepare_data.py --video "大作业数据/数据3-场景.mp4" --out outputs/scene --num_frames 48
python run_vggt.py --scene outputs/scene --device cuda --vggt-repo /path/to/vggt
python ba_optimize.py --scene outputs/scene --iters 1000 --lr_pose 1e-3 --lr_points 1e-2
python train_gaussians.py --scene outputs/scene --mode ba --steps 7000
python evaluate.py --scene outputs/scene
python viewer.py --scene outputs/scene --mode ba
```

Full VGGT and Gaussian training require CUDA plus upstream VGGT and the installed `gsplat` package; CPU environments can run preparation and tests.

## Coding Style & Naming Conventions

Use standard Python formatting with 4-space indentation, explicit imports, and type hints where they clarify data flow. Keep reusable logic in `src/vggt_gaussian_reconstruction/`; keep top-level scripts as thin CLI layers. Name modules and functions in `snake_case`, classes in `PascalCase`, and tests as `test_*.py` with `test_*` functions. Prefer `pathlib.Path` and structured helpers when handling COLMAP files.

## Testing Guidelines

Pytest is configured in `pyproject.toml` with `tests` as the test path and `src` on `PYTHONPATH`. Add focused unit tests for geometry, frame selection, COLMAP conversion, and BA optimization changes. Use `tmp_path` and small synthetic data so tests stay deterministic and fast.

## Commit & Pull Request Guidelines

This repository currently has no commit history to infer a house style from. Use concise imperative commit subjects, for example `Add quality frame selection test` or `Fix COLMAP camera export`. Pull requests should describe the pipeline stage affected, list commands run, and note required external assets such as CUDA, VGGT, `gsplat`, or local video data. Include screenshots or metrics when changing viewer output or reconstruction quality.

## Security & Configuration Tips

Do not commit local datasets, generated reconstructions, checkpoints, or cache files. `.gitignore` already excludes `大作业数据/`, `outputs/`, `__pycache__/`, `.pytest_cache/`, and package metadata. Keep machine-specific paths such as `--vggt-repo` in shell commands or local notes, not source files.
