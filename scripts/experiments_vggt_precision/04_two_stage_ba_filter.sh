#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/share/micromamba/root/envs/VGGT_GSPLAT_A800}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"

export EXP_NAME="${EXP_NAME:-vggt_precision_04_two_stage_ba_filter}"
export SCENE_DIR="${SCENE_DIR:-${REPO_ROOT}/outputs/experiments_vggt_precision/04_two_stage_ba_filter}"
export NUM_FRAMES="${NUM_FRAMES:-200}"
export FRAME_STRATEGY="${FRAME_STRATEGY:-parallax}"
export CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-8}"
export VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---query-frame-num 24 --max-query-pts 12288 --vis-thresh 0.12 --max-reproj-error 6.0 --shared-camera --query-frame-strategy quality}"
export RUN_GSPLAT=0
export RUN_EVAL=0

bash "${REPO_ROOT}/scripts/train_infer_baidu.sh"

export PATH="${ENV_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/filter_colmap_points.py" \
    --input-sparse "${SCENE_DIR}/ba/sparse/0" \
    --output-sparse "${SCENE_DIR}/ba_filtered/sparse/0" \
    --min-track-len "${FILTER_MIN_TRACK_LEN:-3}" \
    --max-error "${FILTER_MAX_ERROR:-3.0}" \
    --report "${SCENE_DIR}/ba_filtered/filter_stats.json"

"${PYTHON_BIN}" "${REPO_ROOT}/ba_optimize.py" \
    --scene "${SCENE_DIR}" \
    --input-sparse "${SCENE_DIR}/ba_filtered/sparse/0" \
    --output-sparse "${SCENE_DIR}/ba/sparse/0"

RUN_PREPARE=0 \
RUN_VGGT=0 \
RUN_BA=0 \
RUN_GSPLAT=1 \
RUN_EVAL=1 \
EXP_NAME="${EXP_NAME}_gsplat" \
GAUSSIAN_STEPS="${GAUSSIAN_STEPS:-30000}" \
GAUSSIAN_DISTRIBUTED="${GAUSSIAN_DISTRIBUTED:-1}" \
bash "${REPO_ROOT}/scripts/train_infer_baidu.sh"
