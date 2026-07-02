# VGGT Precision Experiment Scripts

Run each script as a separate platform job so the experiments do not share `SCENE_DIR` or output names.

```bash
bash scripts/experiments_vggt_precision/00_baseline_quality.sh
bash scripts/experiments_vggt_precision/01_anchor_frame_selection.sh
bash scripts/experiments_vggt_precision/02_parallax_adaptive_query.sh
bash scripts/experiments_vggt_precision/03_coarse_to_fine_frames.sh
bash scripts/experiments_vggt_precision/04_two_stage_ba_filter.sh
```

Override `VIDEO_PATH`, `CVD`, `GAUSSIAN_CVD`, `ENV_PREFIX`, or `REPO_ROOT` the same way as `scripts/train_infer_baidu.sh`.
