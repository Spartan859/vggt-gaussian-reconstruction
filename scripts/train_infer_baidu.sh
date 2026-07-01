#!/usr/bin/env bash
set -euo pipefail

# Baidu training-platform entrypoint for the VGGT Gaussian reconstruction pipeline.
# Defaults use the VGGT_GSPLAT_A800 micromamba environment. Override variables below as needed.

REPO_ROOT="${REPO_ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/share/micromamba/root/envs/VGGT_GSPLAT_A800}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"

VIDEO_PATH="${VIDEO_PATH:-${REPO_ROOT}/大作业数据/数据3-场景.mp4}"
SCENE_DIR="${SCENE_DIR:-${REPO_ROOT}/outputs/scene}"
NUM_FRAMES="${NUM_FRAMES:-48}"
FRAME_STRATEGY="${FRAME_STRATEGY:-quality}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-4}"

CVD="${CVD:-0}"
DEVICE="${DEVICE:-cuda}"
IMPORT_COLMAP="${IMPORT_COLMAP:-}"
USE_VGGT_BA="${USE_VGGT_BA:-0}"
VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:-}"

BA_DEVICE="${BA_DEVICE:-${DEVICE}}"
BA_ITERS="${BA_ITERS:-1000}"
BA_LR_POSE="${BA_LR_POSE:-1e-3}"
BA_LR_POINTS="${BA_LR_POINTS:-1e-2}"
BA_HUBER_DELTA="${BA_HUBER_DELTA:-4.0}"
BA_MIN_TRACK_LEN="${BA_MIN_TRACK_LEN:-2}"

GAUSSIAN_MODE="${GAUSSIAN_MODE:-ba}"
GAUSSIAN_STEPS="${GAUSSIAN_STEPS:-7000}"
GAUSSIAN_LR="${GAUSSIAN_LR:-1e-2}"
GAUSSIAN_IMAGE_SCALE="${GAUSSIAN_IMAGE_SCALE:-1.0}"
GAUSSIAN_MAX_POINTS="${GAUSSIAN_MAX_POINTS:-200000}"
GAUSSIAN_SAVE_EVERY="${GAUSSIAN_SAVE_EVERY:-1000}"
GAUSSIAN_RENDER_EVERY="${GAUSSIAN_RENDER_EVERY:-1000}"

RENDER_DIR_VGGT="${RENDER_DIR_VGGT:-}"
RENDER_DIR_BA="${RENDER_DIR_BA:-}"
VIEWER_COMMAND_TEMPLATE="${VIEWER_COMMAND_TEMPLATE:-}"

TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
VGGT_WEIGHTS_PATH="${VGGT_WEIGHTS_PATH:-${TORCH_HOME}/hub/checkpoints/model.pt}"
VGGT_WEIGHTS_URL="${VGGT_WEIGHTS_URL:-https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_VGGT="${RUN_VGGT:-1}"
RUN_BA="${RUN_BA:-1}"
RUN_GSPLAT="${RUN_GSPLAT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_VIEWER="${RUN_VIEWER:-0}"

STDOUT_LOG_DIR="${STDOUT_LOG_DIR:-${REPO_ROOT}/outputs/platform_logs}"
EXP_NAME="${EXP_NAME:-vggt_gaussian_full_$(date +%Y%m%d_%H%M%S)}"

usage() {
    cat <<'EOF'
Usage: bash scripts/train_infer_baidu.sh [options]

Options:
  --resume-from STAGE   Skip stages before STAGE. Stages: prepare, vggt, ba, gsplat, eval, viewer
  --skip-prepare        Do not extract frames; requires SCENE_DIR/images
  --skip-vggt           Do not run VGGT/COLMAP; requires SCENE_DIR/vggt/sparse/0 with bin or txt COLMAP files
  --skip-ba             Do not run bundle adjustment
  --skip-gsplat         Do not run Gaussian training
  --skip-eval           Do not run evaluation
  --run-viewer          Run viewer command after evaluation
  --skip-viewer         Do not run viewer command
  -h, --help            Show this help

Examples:
  bash scripts/train_infer_baidu.sh --resume-from gsplat
  bash scripts/train_infer_baidu.sh --skip-prepare --skip-vggt
EOF
}

resume_from() {
    case "$1" in
        prepare)
            ;;
        vggt)
            RUN_PREPARE=0
            ;;
        ba)
            RUN_PREPARE=0
            RUN_VGGT=0
            ;;
        gsplat)
            RUN_PREPARE=0
            RUN_VGGT=0
            RUN_BA=0
            ;;
        eval)
            RUN_PREPARE=0
            RUN_VGGT=0
            RUN_BA=0
            RUN_GSPLAT=0
            ;;
        viewer)
            RUN_PREPARE=0
            RUN_VGGT=0
            RUN_BA=0
            RUN_GSPLAT=0
            RUN_EVAL=0
            RUN_VIEWER=1
            ;;
        *)
            echo "Unknown resume stage: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume-from)
            [[ $# -ge 2 ]] || { echo "--resume-from requires a stage" >&2; exit 2; }
            resume_from "$2"
            shift 2
            ;;
        --skip-prepare)
            RUN_PREPARE=0
            shift
            ;;
        --skip-vggt)
            RUN_VGGT=0
            shift
            ;;
        --skip-ba)
            RUN_BA=0
            shift
            ;;
        --skip-gsplat)
            RUN_GSPLAT=0
            shift
            ;;
        --skip-eval)
            RUN_EVAL=0
            shift
            ;;
        --run-viewer)
            RUN_VIEWER=1
            shift
            ;;
        --skip-viewer)
            RUN_VIEWER=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME HF_HOME VGGT_WEIGHTS_PATH VGGT_WEIGHTS_URL
unset PYTHONHOME PYTHONPATH

cd "${REPO_ROOT}"
mkdir -p "${STDOUT_LOG_DIR}"
STDOUT_LOG="${STDOUT_LOG_DIR}/${EXP_NAME}.log"
export STDOUT_LOG
touch "${STDOUT_LOG}"
exec > >(tee -a "${STDOUT_LOG}") 2>&1

log() {
    echo "[$(date '+%F %T')] $*"
}

require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 1
    fi
}

run_step() {
    log "command: CUDA_VISIBLE_DEVICES=${CVD} $*"
    CUDA_VISIBLE_DEVICES="${CVD}" "$@"
}

mkdir -p "${TORCH_HOME}/hub/checkpoints" "${HF_HOME}"

log "repo: ${REPO_ROOT}"
log "scene dir: ${SCENE_DIR}"
log "video: ${VIDEO_PATH}"
log "env python: ${PYTHON_BIN}"
log "torch home: ${TORCH_HOME}"
log "hf home: ${HF_HOME}"
log "vggt weights: ${VGGT_WEIGHTS_PATH}"
log "vggt weights url: ${VGGT_WEIGHTS_URL}"
log "CVD: ${CVD}"
log "device: ${DEVICE}"
log "stdout log: ${STDOUT_LOG}"
log "stages: prepare=${RUN_PREPARE} vggt=${RUN_VGGT} ba=${RUN_BA} gsplat=${RUN_GSPLAT} eval=${RUN_EVAL} viewer=${RUN_VIEWER}"

require_path "${PYTHON_BIN}"
require_path "${REPO_ROOT}/prepare_data.py"
require_path "${REPO_ROOT}/run_vggt.py"
require_path "${REPO_ROOT}/ba_optimize.py"
require_path "${REPO_ROOT}/train_gaussians.py"
require_path "${REPO_ROOT}/evaluate.py"

"${PYTHON_BIN}" - <<'PY'
import sys
print("python", sys.version.replace("\n", " "))
PY

log "checking python dependencies"
"${PYTHON_BIN}" - <<'PY'
import importlib
missing = []
for name in ("numpy", "PIL", "torch", "tqdm"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, repr(exc)))
if missing:
    for name, exc in missing:
        print(f"missing {name}: {exc}")
    raise SystemExit(1)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

if [[ "${RUN_PREPARE}" == "1" ]]; then
    log "starting frame extraction"
    require_path "${VIDEO_PATH}"
    run_step "${PYTHON_BIN}" prepare_data.py \
        --video "${VIDEO_PATH}" \
        --out "${SCENE_DIR}" \
        --num_frames "${NUM_FRAMES}" \
        --strategy "${FRAME_STRATEGY}" \
        --candidate_multiplier "${CANDIDATE_MULTIPLIER}"
else
    log "skipping frame extraction"
    require_path "${SCENE_DIR}/images"
fi

if [[ "${RUN_VGGT}" == "1" ]]; then
    log "starting VGGT/COLMAP initialization"
    VGGT_CMD=("${PYTHON_BIN}" run_vggt.py --scene "${SCENE_DIR}" --device "${DEVICE}")
    if [[ -n "${IMPORT_COLMAP}" ]]; then
        require_path "${IMPORT_COLMAP}"
        VGGT_CMD+=(--import-colmap "${IMPORT_COLMAP}")
    else
        if [[ "${USE_VGGT_BA}" == "1" ]]; then
            VGGT_CMD+=(--use-ba)
        fi
        if [[ -n "${VGGT_EXTRA_ARGS}" ]]; then
            # shellcheck disable=SC2206
            VGGT_EXTRA_ARGS_ARRAY=(${VGGT_EXTRA_ARGS})
            VGGT_CMD+=(--extra-args "${VGGT_EXTRA_ARGS_ARRAY[@]}")
        fi
    fi
    run_step "${VGGT_CMD[@]}"
else
    log "skipping VGGT/COLMAP initialization"
    if [[ ! -e "${SCENE_DIR}/vggt/sparse/0/cameras.txt" && ! -e "${SCENE_DIR}/vggt/sparse/0/cameras.bin" ]]; then
        echo "Missing COLMAP cameras file in: ${SCENE_DIR}/vggt/sparse/0" >&2
        exit 1
    fi
    if [[ ! -e "${SCENE_DIR}/vggt/sparse/0/images.txt" && ! -e "${SCENE_DIR}/vggt/sparse/0/images.bin" ]]; then
        echo "Missing COLMAP images file in: ${SCENE_DIR}/vggt/sparse/0" >&2
        exit 1
    fi
fi

if [[ "${RUN_BA}" == "1" ]]; then
    log "starting bundle adjustment"
    run_step "${PYTHON_BIN}" ba_optimize.py \
        --scene "${SCENE_DIR}" \
        --iters "${BA_ITERS}" \
        --lr_pose "${BA_LR_POSE}" \
        --lr_points "${BA_LR_POINTS}" \
        --huber_delta "${BA_HUBER_DELTA}" \
        --min_track_len "${BA_MIN_TRACK_LEN}" \
        --device "${BA_DEVICE}"
else
    log "skipping bundle adjustment"
fi

if [[ "${RUN_GSPLAT}" == "1" ]]; then
    log "starting Gaussian training"
    GSPLAT_CMD=(
        "${PYTHON_BIN}" train_gaussians.py
        --scene "${SCENE_DIR}"
        --mode "${GAUSSIAN_MODE}"
        --steps "${GAUSSIAN_STEPS}"
        --device "${DEVICE}"
        --lr "${GAUSSIAN_LR}"
        --image-scale "${GAUSSIAN_IMAGE_SCALE}"
        --max-points "${GAUSSIAN_MAX_POINTS}"
        --save-every "${GAUSSIAN_SAVE_EVERY}"
        --render-every "${GAUSSIAN_RENDER_EVERY}"
    )
    run_step "${GSPLAT_CMD[@]}"
else
    log "skipping Gaussian training"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
    log "starting evaluation summary"
    EVAL_CMD=("${PYTHON_BIN}" evaluate.py --scene "${SCENE_DIR}")
    if [[ -n "${RENDER_DIR_VGGT}" ]]; then
        require_path "${RENDER_DIR_VGGT}"
        EVAL_CMD+=(--render-dir-vggt "${RENDER_DIR_VGGT}")
    fi
    if [[ -n "${RENDER_DIR_BA}" ]]; then
        require_path "${RENDER_DIR_BA}"
        EVAL_CMD+=(--render-dir-ba "${RENDER_DIR_BA}")
    fi
    run_step "${EVAL_CMD[@]}"
else
    log "skipping evaluation summary"
fi

if [[ "${RUN_VIEWER}" == "1" ]]; then
    log "starting viewer command"
    if [[ -z "${VIEWER_COMMAND_TEMPLATE}" ]]; then
        echo "Set VIEWER_COMMAND_TEMPLATE before RUN_VIEWER=1." >&2
        exit 1
    fi
    run_step "${PYTHON_BIN}" viewer.py \
        --scene "${SCENE_DIR}" \
        --mode "${GAUSSIAN_MODE}" \
        --command-template "${VIEWER_COMMAND_TEMPLATE}"
fi

log "pipeline finished"
log "outputs: ${SCENE_DIR}"
