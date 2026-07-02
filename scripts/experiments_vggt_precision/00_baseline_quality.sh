#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"

export EXP_NAME="${EXP_NAME:-vggt_precision_00_baseline_quality}"
export SCENE_DIR="${SCENE_DIR:-${REPO_ROOT}/outputs/experiments_vggt_precision/00_baseline_quality}"
export NUM_FRAMES="${NUM_FRAMES:-160}"
export FRAME_STRATEGY="${FRAME_STRATEGY:-quality}"
export CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-6}"
export VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---query-frame-num 16 --max-query-pts 8192 --vis-thresh 0.15 --max-reproj-error 6.0 --shared-camera --query-frame-strategy linspace}"
export BA_MIN_TRACK_LEN="${BA_MIN_TRACK_LEN:-2}"
export GAUSSIAN_STEPS="${GAUSSIAN_STEPS:-30000}"
export GAUSSIAN_DISTRIBUTED="${GAUSSIAN_DISTRIBUTED:-1}"

bash "${REPO_ROOT}/scripts/train_infer_baidu.sh"
