# VGGT Precision Experiments

This document records experiments for improving VGGT-driven reconstruction precision. The first batch excludes dynamic-object masks and focuses on frame selection, query-frame selection, coarse-to-fine input scaling, and two-stage BA filtering.

## Common Evaluation

Compare each run using:

- `metadata.json`: selected frame count, strategy, candidate scores.
- `vggt/sparse/0` summary: cameras, points, observations, average track length.
- `ba/ba_stats.json`: points and observations before/after BA.
- `runs/<exp>/eval_report.json`: image metrics and rendered outputs when available.
- Visual inspection in `viewer.py`: camera path stability, floaters, point cloud completeness, Gaussian artifacts.

Use the same input video, environment, Gaussian training steps, and viewer filtering settings across experiments.

## 00 Baseline Quality

Script: `scripts/experiments_vggt_precision/00_baseline_quality.sh`

Purpose: establish a strong baseline using the existing `quality` frame selector with more frames and denser VGGT tracks.

Key settings:

```bash
NUM_FRAMES=160
FRAME_STRATEGY=quality
VGGT_EXTRA_ARGS="--query-frame-num 16 --max-query-pts 8192 --vis-thresh 0.15 --max-reproj-error 6.0 --shared-camera"
```

Expected result: better coverage than the old 48/96-frame defaults, but query frames are still uniform in time.

## 01 Anchor Frame Selection

Script: `scripts/experiments_vggt_precision/01_anchor_frame_selection.sh`

Purpose: test whether choosing a stable geometric anchor improves global pose/depth consistency. The selector scores frames by sharpness, exposure, temporal centrality, and appearance novelty; the best anchor is selected first.

Key settings:

```bash
FRAME_STRATEGY=anchor
VGGT_EXTRA_ARGS="--query-frame-strategy anchor --query-frame-num 16 --max-query-pts 8192 ..."
```

Record whether BA produces fewer outlier points and whether the camera path is less warped.

## 02 Parallax + Adaptive Query Frames

Script: `scripts/experiments_vggt_precision/02_parallax_adaptive_query.sh`

Purpose: favor frames with stronger appearance/parallax changes while keeping enough overlap, then rank VGGSfM query frames by quality and novelty instead of fixed `linspace`.

Key settings:

```bash
NUM_FRAMES=200
FRAME_STRATEGY=parallax
VGGT_EXTRA_ARGS="--query-frame-strategy quality --query-frame-num 24 --max-query-pts 12288 --vis-thresh 0.12 --max-reproj-error 5.0 --shared-camera"
```

Expected result: more long tracks and better surface coverage. Watch for failure cases where excessive viewpoint change reduces overlap.

## 03 Coarse-To-Fine Frame Scaling

Script: `scripts/experiments_vggt_precision/03_coarse_to_fine_frames.sh`

Purpose: compare a stable 80-frame anchor reconstruction against a 240-frame parallax reconstruction. This is the first practical step toward chunked coarse-to-fine VGGT.

Current implementation runs two independent reconstructions:

- `03_coarse_to_fine_coarse`: 80 anchor frames, lighter tracks.
- `03_coarse_to_fine_fine`: 240 parallax frames, denser tracks.

Analysis goal: determine whether more frames improve BA/Gaussian quality or mainly introduce noisy tracks. If fine improves geometry, the next step is merging coarse camera priors with fine local tracks.

## 04 Two-Stage BA Filtering

Script: `scripts/experiments_vggt_precision/04_two_stage_ba_filter.sh`

Purpose: run VGGT and BA once, remove weak 3D points, then rerun BA and Gaussian training from the filtered model.

Key settings:

```bash
FILTER_MIN_TRACK_LEN=3
FILTER_MAX_ERROR=3.0
```

Expected result: fewer floating points and cleaner Gaussian initialization. If the scene becomes incomplete, relax `FILTER_MAX_ERROR` to `4.0` or `FILTER_MIN_TRACK_LEN` to `2`.

## Experiment Notes

| Experiment | Date | Job ID | Points / Obs after BA | Eval PSNR/SSIM/LPIPS | Visual result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 00 baseline quality |  |  |  |  |  |  |
| 01 anchor selection |  |  |  |  |  |  |
| 02 parallax adaptive query |  |  |  |  |  |  |
| 03 coarse |  |  |  |  |  |  |
| 03 fine |  |  |  |  |  |  |
| 04 two-stage BA filter |  |  |  |  |  |  |
