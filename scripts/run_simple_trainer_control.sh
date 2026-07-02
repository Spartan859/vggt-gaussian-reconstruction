#!/usr/bin/env bash
set -euo pipefail

# Run external/gsplat's official examples/simple_trainer.py as a control group.
# This does not use this repository's Gaussian trainer.

REPO_ROOT="${REPO_ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/share/micromamba/root/envs/VGGT_GSPLAT_A800}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"

SCENE_DIR="${SCENE_DIR:-${REPO_ROOT}/outputs/scene}"
GSPLAT_REPO="${GSPLAT_REPO:-${REPO_ROOT}/external/gsplat}"
SIMPLE_MODE="${SIMPLE_MODE:-ba}"
SIMPLE_CONFIG="${SIMPLE_CONFIG:-default}"
SIMPLE_TEST_EVERY="${SIMPLE_TEST_EVERY:-0}"
SIMPLE_STEPS_SCALER="${SIMPLE_STEPS_SCALER:-1.0}"
SIMPLE_MAX_STEPS="${SIMPLE_MAX_STEPS:-30000}"
SIMPLE_DISABLE_VIDEO="${SIMPLE_DISABLE_VIDEO:-1}"
SIMPLE_DATA_DIR="${SIMPLE_DATA_DIR:-${SCENE_DIR}/simple_trainer_data_${SIMPLE_MODE}}"

EXP_NAME="${EXP_NAME:-simple_trainer_${SIMPLE_MODE}_$(date +%Y%m%d_%H%M%S)}"
SIMPLE_RESULT_DIR="${SIMPLE_RESULT_DIR:-${SCENE_DIR}/runs/${EXP_NAME}}"
STDOUT_LOG_DIR="${STDOUT_LOG_DIR:-${REPO_ROOT}/outputs/platform_logs}"
STDOUT_LOG="${STDOUT_LOG:-${STDOUT_LOG_DIR}/${EXP_NAME}.log}"

CVD="${CVD:-0}"
SIMPLE_EXTRA_ARGS="${SIMPLE_EXTRA_ARGS:-}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

log() {
    printf '[%(%F %T)T] %s\n' -1 "$*"
}

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "Missing required path: $1" >&2
        exit 1
    fi
}

prepare_simple_data_dir() {
    local sparse_src="${SCENE_DIR}/${SIMPLE_MODE}/sparse/0"
    local sparse_dst="${SIMPLE_DATA_DIR}/sparse/0"

    require_path "${SCENE_DIR}/images"
    require_path "${sparse_src}"

    mkdir -p "${SIMPLE_DATA_DIR}" "${SIMPLE_DATA_DIR}/sparse"
    rm -f "${SIMPLE_DATA_DIR}/images" "${SIMPLE_DATA_DIR}/sparse/0"
    ln -s "${SCENE_DIR}/images" "${SIMPLE_DATA_DIR}/images"
    ln -s "${sparse_src}" "${sparse_dst}"
}

usage() {
    cat <<'EOF'
Usage: bash scripts/run_simple_trainer_control.sh [--help]

Environment variables:
  SIMPLE_MODE         Sparse model under SCENE_DIR to use: ba or vggt. Default: ba
  SIMPLE_RESULT_DIR   Output directory. Default: SCENE_DIR/runs/$EXP_NAME
  SIMPLE_MAX_STEPS    simple_trainer max_steps. Default: 30000
  SIMPLE_TEST_EVERY   Every N images held out for validation. 0 trains all images.
  SIMPLE_DISABLE_VIDEO Disable trajectory video rendering. Default: 1
  CVD                 CUDA_VISIBLE_DEVICES. Use 0,1,2,3,4,5,6,7 for 8 GPUs.
  SIMPLE_EXTRA_ARGS   Extra raw arguments appended to simple_trainer.py.

Examples:
  bash scripts/run_simple_trainer_control.sh
  CVD=0,1,2,3,4,5,6,7 SIMPLE_STEPS_SCALER=0.125 bash scripts/run_simple_trainer_control.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p "${STDOUT_LOG_DIR}"
touch "${STDOUT_LOG}"
exec > >(tee -a "${STDOUT_LOG}") 2>&1
mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

require_path "${PYTHON_BIN}"
require_path "${GSPLAT_REPO}/examples/simple_trainer.py"
prepare_simple_data_dir
mkdir -p "${SIMPLE_RESULT_DIR}"

CMD=(
    "${PYTHON_BIN}" "${GSPLAT_REPO}/examples/simple_trainer.py"
    "${SIMPLE_CONFIG}"
    --data_dir "${SIMPLE_DATA_DIR}"
    --result_dir "${SIMPLE_RESULT_DIR}"
    --data_factor 1
    --test_every "${SIMPLE_TEST_EVERY}"
    --steps_scaler "${SIMPLE_STEPS_SCALER}"
    --max_steps "${SIMPLE_MAX_STEPS}"
    --disable_viewer
)

if [[ "${SIMPLE_DISABLE_VIDEO}" == "1" ]]; then
    CMD+=(--disable_video)
fi

if [[ -n "${SIMPLE_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(${SIMPLE_EXTRA_ARGS})
    CMD+=("${EXTRA_ARGS[@]}")
fi

log "repo: ${REPO_ROOT}"
log "scene: ${SCENE_DIR}"
log "simple data dir: ${SIMPLE_DATA_DIR}"
log "simple result dir: ${SIMPLE_RESULT_DIR}"
log "stdout log: ${STDOUT_LOG}"
log "gsplat repo: ${GSPLAT_REPO}"
log "MPLCONFIGDIR: ${MPLCONFIGDIR}"
log "SIMPLE_DISABLE_VIDEO: ${SIMPLE_DISABLE_VIDEO}"
log "command: CUDA_VISIBLE_DEVICES=${CVD} PYTHONPATH=${GSPLAT_REPO}:${GSPLAT_REPO}/examples ${CMD[*]}"

CUDA_VISIBLE_DEVICES="${CVD}" \
PYTHONPATH="${GSPLAT_REPO}:${GSPLAT_REPO}/examples:${PYTHONPATH:-}" \
"${CMD[@]}"
