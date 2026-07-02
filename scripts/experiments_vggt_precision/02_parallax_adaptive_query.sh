#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"

export EXP_NAME="${EXP_NAME:-vggt_precision_02_parallax_adaptive_query}"
export SCENE_DIR="${SCENE_DIR:-${REPO_ROOT}/outputs/experiments_vggt_precision/02_parallax_adaptive_query}"
export NUM_FRAMES="${NUM_FRAMES:-200}"
export FRAME_STRATEGY="${FRAME_STRATEGY:-parallax}"
export CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-8}"
export VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---query-frame-num 24 --max-query-pts 12288 --vis-thresh 0.12 --max-reproj-error 5.0 --shared-camera --query-frame-strategy quality}"
export BA_MIN_TRACK_LEN="${BA_MIN_TRACK_LEN:-2}"
export GAUSSIAN_STEPS="${GAUSSIAN_STEPS:-30000}"
export GAUSSIAN_DISTRIBUTED="${GAUSSIAN_DISTRIBUTED:-1}"

bash "${REPO_ROOT}/scripts/train_infer_baidu.sh"
