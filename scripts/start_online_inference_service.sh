#!/usr/bin/env bash
set -euo pipefail

# Startup command for Baidu AIHC online services.
# It launches the browser-based real-time gsplat viewer and listens on one HTTP port.

REPO_ROOT="${REPO_ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/share/micromamba/root/envs/VGGT_GSPLAT_A800}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"

SCENE_DIR="${SCENE_DIR:-${REPO_ROOT}/outputs/scene}"
GAUSSIAN_MODE="${GAUSSIAN_MODE:-ba}"
RUN_DIR="${RUN_DIR:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DEVICE="${DEVICE:-cuda}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
DISPLAY_HOST="${DISPLAY_HOST:-}"
VIEWER_PORT="${VIEWER_PORT:-${PORT:-${SERVICE_PORT:-${AIHC_SERVICE_PORT:-8080}}}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "Missing required path: $1" >&2
        exit 1
    fi
}

case "${VIEWER_PORT}" in
    ''|*[!0-9]*)
        echo "VIEWER_PORT must be an integer, got: ${VIEWER_PORT}" >&2
        exit 2
        ;;
esac

if [[ "${VIEWER_PORT}" == "8001" || "${VIEWER_PORT}" == "8002" ]]; then
    echo "Do not use port ${VIEWER_PORT}; AIHC reserves 8001 and 8002." >&2
    exit 2
fi

require_path "${PYTHON_BIN}"
require_path "${REPO_ROOT}/viewer.py"
require_path "${SCENE_DIR}"

VIEWER_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/viewer.py"
    --scene "${SCENE_DIR}"
    --mode "${GAUSSIAN_MODE}"
    --device "${DEVICE}"
    --port "${VIEWER_PORT}"
    --bind-host "${BIND_HOST}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
    require_path "${CHECKPOINT_PATH}"
    VIEWER_CMD+=(--checkpoint "${CHECKPOINT_PATH}")
elif [[ -n "${RUN_DIR}" ]]; then
    require_path "${RUN_DIR}"
    VIEWER_CMD+=(--run-dir "${RUN_DIR}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
    VIEWER_CMD+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${DISPLAY_HOST}" ]]; then
    VIEWER_CMD+=(--host "${DISPLAY_HOST}")
fi

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

log "repo: ${REPO_ROOT}"
log "scene: ${SCENE_DIR}"
log "mode: ${GAUSSIAN_MODE}"
log "device: ${DEVICE}"
log "listen: ${BIND_HOST}:${VIEWER_PORT}"
if [[ -n "${CHECKPOINT_PATH}" ]]; then
    log "checkpoint: ${CHECKPOINT_PATH}"
elif [[ -n "${RUN_DIR}" ]]; then
    log "run dir: ${RUN_DIR}"
else
    log "checkpoint: newest under ${SCENE_DIR}/runs/*/gaussians_${GAUSSIAN_MODE}/checkpoint.pt"
fi
log "command: ${VIEWER_CMD[*]}"

cd "${REPO_ROOT}"
exec "${VIEWER_CMD[@]}"
