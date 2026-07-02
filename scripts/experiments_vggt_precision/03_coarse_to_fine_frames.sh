#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"

COARSE_SCENE_DIR="${COARSE_SCENE_DIR:-${REPO_ROOT}/outputs/experiments_vggt_precision/03_coarse_to_fine_coarse}"
FINE_SCENE_DIR="${FINE_SCENE_DIR:-${REPO_ROOT}/outputs/experiments_vggt_precision/03_coarse_to_fine_fine}"

EXP_NAME="${EXP_NAME:-vggt_precision_03a_coarse}" \
SCENE_DIR="${COARSE_SCENE_DIR}" \
NUM_FRAMES="${COARSE_NUM_FRAMES:-80}" \
FRAME_STRATEGY="${COARSE_FRAME_STRATEGY:-anchor}" \
CANDIDATE_MULTIPLIER="${COARSE_CANDIDATE_MULTIPLIER:-8}" \
VGGT_EXTRA_ARGS="${COARSE_VGGT_EXTRA_ARGS:---query-frame-num 12 --max-query-pts 8192 --vis-thresh 0.15 --max-reproj-error 6.0 --shared-camera --query-frame-strategy anchor}" \
GAUSSIAN_STEPS="${COARSE_GAUSSIAN_STEPS:-12000}" \
GAUSSIAN_DISTRIBUTED="${GAUSSIAN_DISTRIBUTED:-1}" \
bash "${REPO_ROOT}/scripts/train_infer_baidu.sh"

EXP_NAME="${FINE_EXP_NAME:-vggt_precision_03b_fine}" \
SCENE_DIR="${FINE_SCENE_DIR}" \
NUM_FRAMES="${FINE_NUM_FRAMES:-240}" \
FRAME_STRATEGY="${FINE_FRAME_STRATEGY:-parallax}" \
CANDIDATE_MULTIPLIER="${FINE_CANDIDATE_MULTIPLIER:-8}" \
VGGT_EXTRA_ARGS="${FINE_VGGT_EXTRA_ARGS:---query-frame-num 24 --max-query-pts 12288 --vis-thresh 0.12 --max-reproj-error 5.0 --shared-camera --query-frame-strategy quality}" \
GAUSSIAN_STEPS="${FINE_GAUSSIAN_STEPS:-30000}" \
GAUSSIAN_DISTRIBUTED="${GAUSSIAN_DISTRIBUTED:-1}" \
bash "${REPO_ROOT}/scripts/train_infer_baidu.sh"
