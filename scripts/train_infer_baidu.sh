#!/usr/bin/env bash
set -euo pipefail

# Baidu training-platform entrypoint for the VGGT Gaussian reconstruction pipeline.
# Defaults use the VGGT_GSPLAT_A800 micromamba environment. Override variables below as needed.

REPO_ROOT="${REPO_ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/share/micromamba/root/envs/VGGT_GSPLAT_A800}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"

VIDEO_PATH="${VIDEO_PATH:-${REPO_ROOT}/大作业数据/数据3-场景.mp4}"
SCENE_DIR="${SCENE_DIR:-${REPO_ROOT}/outputs/scene}"
NUM_FRAMES="${NUM_FRAMES:-96}"
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
GAUSSIAN_STEPS="${GAUSSIAN_STEPS:-30000}"
GAUSSIAN_LR="${GAUSSIAN_LR:-1.0}"
GAUSSIAN_IMAGE_SCALE="${GAUSSIAN_IMAGE_SCALE:-1.0}"
GAUSSIAN_MAX_POINTS="${GAUSSIAN_MAX_POINTS:-200000}"
GAUSSIAN_MAX_GAUSSIANS="${GAUSSIAN_MAX_GAUSSIANS:-200000}"
GAUSSIAN_TEST_EVERY="${GAUSSIAN_TEST_EVERY:-8}"
GAUSSIAN_REFINE_STOP_ITER="${GAUSSIAN_REFINE_STOP_ITER:-15000}"
GAUSSIAN_GROW_GRAD2D="${GAUSSIAN_GROW_GRAD2D:-0.0002}"
GAUSSIAN_PRUNE_OPA="${GAUSSIAN_PRUNE_OPA:-0.01}"
GAUSSIAN_MIN_TRACK_LEN="${GAUSSIAN_MIN_TRACK_LEN:-4}"
GAUSSIAN_OPACITY_REG="${GAUSSIAN_OPACITY_REG:-0.001}"
GAUSSIAN_SCALE_REG="${GAUSSIAN_SCALE_REG:-0.001}"
GAUSSIAN_PRUNE_EVERY="${GAUSSIAN_PRUNE_EVERY:-1000}"
GAUSSIAN_PRUNE_LARGE_SCALE="${GAUSSIAN_PRUNE_LARGE_SCALE:-0.25}"
GAUSSIAN_VISIBILITY_PRUNE_EVERY="${GAUSSIAN_VISIBILITY_PRUNE_EVERY:-2000}"
GAUSSIAN_VISIBILITY_PRUNE_START="${GAUSSIAN_VISIBILITY_PRUNE_START:-8000}"
GAUSSIAN_VISIBILITY_MIN_VIEWS="${GAUSSIAN_VISIBILITY_MIN_VIEWS:-2}"
GAUSSIAN_PRUNE_SCENE_RADIUS="${GAUSSIAN_PRUNE_SCENE_RADIUS:-2.5}"
GAUSSIAN_DEPTH_LOSS="${GAUSSIAN_DEPTH_LOSS:-1}"
GAUSSIAN_DEPTH_LAMBDA="${GAUSSIAN_DEPTH_LAMBDA:-0.01}"
GAUSSIAN_DEPTH_SAMPLE_COUNT="${GAUSSIAN_DEPTH_SAMPLE_COUNT:-2048}"
GAUSSIAN_DEPTH_LOSS_CLAMP="${GAUSSIAN_DEPTH_LOSS_CLAMP:-2.0}"
GAUSSIAN_MASK_LOSS="${GAUSSIAN_MASK_LOSS:-1}"
GAUSSIAN_MASK_DIR="${GAUSSIAN_MASK_DIR:-${SCENE_DIR}/masks}"
GAUSSIAN_MASK_ALPHA_LAMBDA="${GAUSSIAN_MASK_ALPHA_LAMBDA:-0.05}"
GAUSSIAN_MASK_THRESHOLD="${GAUSSIAN_MASK_THRESHOLD:-0.5}"
GAUSSIAN_OPACITIES_LR="${GAUSSIAN_OPACITIES_LR:-0.02}"
GAUSSIAN_OPACITY_RESET_EVERY="${GAUSSIAN_OPACITY_RESET_EVERY:-1000}"
GAUSSIAN_OPACITY_RESET_UNTIL="${GAUSSIAN_OPACITY_RESET_UNTIL:-0}"
GAUSSIAN_FINAL_PRUNE_OPA="${GAUSSIAN_FINAL_PRUNE_OPA:-0.05}"
GAUSSIAN_FINAL_PRUNE_LARGE_SCALE="${GAUSSIAN_FINAL_PRUNE_LARGE_SCALE:-0.08}"
GAUSSIAN_FINAL_PRUNE_SCENE_RADIUS="${GAUSSIAN_FINAL_PRUNE_SCENE_RADIUS:-2.8}"
GAUSSIAN_FINAL_VISIBILITY_MIN_VIEWS="${GAUSSIAN_FINAL_VISIBILITY_MIN_VIEWS:-3}"
GAUSSIAN_DISTRIBUTED="${GAUSSIAN_DISTRIBUTED:-0}"
if [[ -z "${GAUSSIAN_CVD:-}" ]]; then
    if [[ "${GAUSSIAN_DISTRIBUTED}" == "1" ]]; then
        GAUSSIAN_CVD="0,1,2,3,4,5,6,7"
    else
        GAUSSIAN_CVD="${CVD}"
    fi
fi
GAUSSIAN_SAVE_EVERY="${GAUSSIAN_SAVE_EVERY:-1000}"
GAUSSIAN_RENDER_EVERY="${GAUSSIAN_RENDER_EVERY:-1000}"

RENDER_DIR_VGGT="${RENDER_DIR_VGGT:-}"
RENDER_DIR_BA="${RENDER_DIR_BA:-}"
VIEWER_PORT="${VIEWER_PORT:-8080}"

TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
VGGT_WEIGHTS_PATH="${VGGT_WEIGHTS_PATH:-${TORCH_HOME}/hub/checkpoints/model.pt}"
VGGT_WEIGHTS_URL="${VGGT_WEIGHTS_URL:-https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt}"
VGGSFM_TRACKER_WEIGHTS_PATH="${VGGSFM_TRACKER_WEIGHTS_PATH:-${TORCH_HOME}/hub/checkpoints/vggsfm_v2_tracker.pt}"
VGGSFM_TRACKER_WEIGHTS_URL="${VGGSFM_TRACKER_WEIGHTS_URL:-https://hf-mirror.com/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt}"
ALIKED_WEIGHTS_PATH="${ALIKED_WEIGHTS_PATH:-${TORCH_HOME}/hub/checkpoints/aliked-n16.pth}"
ALIKED_WEIGHTS_URL="${ALIKED_WEIGHTS_URL:-https://gh-proxy.org/https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth}"
SUPERPOINT_WEIGHTS_PATH="${SUPERPOINT_WEIGHTS_PATH:-${TORCH_HOME}/hub/checkpoints/superpoint_v1.pth}"
SUPERPOINT_WEIGHTS_URL="${SUPERPOINT_WEIGHTS_URL:-https://gh-proxy.org/https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_VGGT="${RUN_VGGT:-1}"
RUN_BA="${RUN_BA:-1}"
RUN_GSPLAT="${RUN_GSPLAT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_VIEWER="${RUN_VIEWER:-0}"

STDOUT_LOG_DIR="${STDOUT_LOG_DIR:-${REPO_ROOT}/outputs/platform_logs}"
if [[ -z "${EXP_NAME:-}" ]]; then
    EXP_NAME="vggt_gaussian_full_$(date +%Y%m%d_%H%M%S)"
fi
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${SCENE_DIR}/runs/${EXP_NAME}}"
GAUSSIAN_OUTPUT_DIR="${GAUSSIAN_OUTPUT_DIR:-${RUN_OUTPUT_DIR}/gaussians_${GAUSSIAN_MODE}}"
EVAL_REPORT_PATH="${EVAL_REPORT_PATH:-${RUN_OUTPUT_DIR}/eval_report.json}"

usage() {
    cat <<'EOF'
Usage: bash scripts/train_infer_baidu.sh [options]

Options:
  --resume-from STAGE   Skip stages before STAGE. Stages: prepare, vggt, ba, gsplat, eval, viewer
  --skip-prepare        Do not extract frames; requires SCENE_DIR/images
  --skip-vggt           Do not run VGGT/COLMAP; requires SCENE_DIR/vggt/sparse/0 with bin or txt COLMAP files
                        If that sparse model has no BA-ready observations, VGGT will be rerun.
  --skip-ba             Do not run bundle adjustment
  --skip-gsplat         Do not run Gaussian training
  --skip-eval           Do not run evaluation
  --run-viewer          Run viewer command after evaluation
  --skip-viewer         Do not run viewer command
  -h, --help            Show this help

Output:
  Each run writes Gaussian outputs and eval_report.json under:
    ${SCENE_DIR}/runs/${EXP_NAME}/
  Override RUN_OUTPUT_DIR, GAUSSIAN_OUTPUT_DIR, or EVAL_REPORT_PATH if needed.
  Set VIEWER_PORT to choose the browser viewer port when --run-viewer is used.

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
export TORCH_HOME HF_HOME VGGT_WEIGHTS_PATH VGGT_WEIGHTS_URL VGGSFM_TRACKER_WEIGHTS_PATH VGGSFM_TRACKER_WEIGHTS_URL
export ALIKED_WEIGHTS_PATH ALIKED_WEIGHTS_URL SUPERPOINT_WEIGHTS_PATH SUPERPOINT_WEIGHTS_URL
unset PYTHONHOME PYTHONPATH

cd "${REPO_ROOT}"
mkdir -p "${STDOUT_LOG_DIR}" "${RUN_OUTPUT_DIR}"
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

is_ba_ready_sparse() {
    local sparse_dir="$1"
    local min_track_len="$2"
    "${PYTHON_BIN}" - "$sparse_dir" "$min_track_len" <<'PY'
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path.cwd() / "src"))
from vggt_gaussian_reconstruction.colmap import read_model

sparse_dir = Path(sys.argv[1])
min_track_len = int(sys.argv[2])
try:
    model = read_model(sparse_dir)
except Exception:
    raise SystemExit(1)

usable_points = [
    point
    for point in model.points3d.values()
    if len(point.track) >= min_track_len and np.all(np.isfinite(point.xyz))
]
observations = sum(len(point.track) for point in usable_points)
raise SystemExit(0 if usable_points and observations > 0 else 1)
PY
}

run_step() {
    log "command: CUDA_VISIBLE_DEVICES=${CVD} $*"
    CUDA_VISIBLE_DEVICES="${CVD}" "$@"
}

run_step_with_cvd() {
    local cvd="$1"
    shift
    log "command: CUDA_VISIBLE_DEVICES=${cvd} $*"
    CUDA_VISIBLE_DEVICES="${cvd}" "$@"
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
log "vggsfm tracker weights: ${VGGSFM_TRACKER_WEIGHTS_PATH}"
log "vggsfm tracker weights url: ${VGGSFM_TRACKER_WEIGHTS_URL}"
log "aliked weights: ${ALIKED_WEIGHTS_PATH}"
log "superpoint weights: ${SUPERPOINT_WEIGHTS_PATH}"
log "CVD: ${CVD}"
log "gaussian CVD: ${GAUSSIAN_CVD}"
log "gaussian distributed: ${GAUSSIAN_DISTRIBUTED}"
log "gaussian mask loss: ${GAUSSIAN_MASK_LOSS}"
log "gaussian mask dir: ${GAUSSIAN_MASK_DIR}"
log "device: ${DEVICE}"
log "stdout log: ${STDOUT_LOG}"
log "run output dir: ${RUN_OUTPUT_DIR}"
log "gaussian output dir: ${GAUSSIAN_OUTPUT_DIR}"
log "eval report path: ${EVAL_REPORT_PATH}"
log "viewer port: ${VIEWER_PORT}"
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

if [[ "${RUN_BA}" == "1" && "${RUN_VGGT}" == "0" ]]; then
    if ! is_ba_ready_sparse "${SCENE_DIR}/vggt/sparse/0" "${BA_MIN_TRACK_LEN}"; then
        log "existing sparse model is missing or not BA-ready; forcing VGGT stage"
        RUN_VGGT=1
    fi
fi

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
            VGGT_CMD+=("${VGGT_EXTRA_ARGS_ARRAY[@]}")
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
    log "starting pycolmap bundle adjustment"
    run_step "${PYTHON_BIN}" ba_optimize.py \
        --scene "${SCENE_DIR}"
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
        --max-gaussians "${GAUSSIAN_MAX_GAUSSIANS}"
        --opacities-lr "${GAUSSIAN_OPACITIES_LR}"
        --test-every "${GAUSSIAN_TEST_EVERY}"
        --refine-stop-iter "${GAUSSIAN_REFINE_STOP_ITER}"
        --grow-grad2d "${GAUSSIAN_GROW_GRAD2D}"
        --prune-opa "${GAUSSIAN_PRUNE_OPA}"
        --min-track-len "${GAUSSIAN_MIN_TRACK_LEN}"
        --opacity-reg "${GAUSSIAN_OPACITY_REG}"
        --scale-reg "${GAUSSIAN_SCALE_REG}"
        --prune-every "${GAUSSIAN_PRUNE_EVERY}"
        --prune-large-scale "${GAUSSIAN_PRUNE_LARGE_SCALE}"
        --visibility-prune-every "${GAUSSIAN_VISIBILITY_PRUNE_EVERY}"
        --visibility-prune-start "${GAUSSIAN_VISIBILITY_PRUNE_START}"
        --visibility-min-views "${GAUSSIAN_VISIBILITY_MIN_VIEWS}"
        --prune-scene-radius "${GAUSSIAN_PRUNE_SCENE_RADIUS}"
        --depth-lambda "${GAUSSIAN_DEPTH_LAMBDA}"
        --depth-sample-count "${GAUSSIAN_DEPTH_SAMPLE_COUNT}"
        --depth-loss-clamp "${GAUSSIAN_DEPTH_LOSS_CLAMP}"
        --mask-alpha-lambda "${GAUSSIAN_MASK_ALPHA_LAMBDA}"
        --mask-threshold "${GAUSSIAN_MASK_THRESHOLD}"
        --opacity-reset-every "${GAUSSIAN_OPACITY_RESET_EVERY}"
        --opacity-reset-until "${GAUSSIAN_OPACITY_RESET_UNTIL}"
        --final-prune-opa "${GAUSSIAN_FINAL_PRUNE_OPA}"
        --final-prune-large-scale "${GAUSSIAN_FINAL_PRUNE_LARGE_SCALE}"
        --final-prune-scene-radius "${GAUSSIAN_FINAL_PRUNE_SCENE_RADIUS}"
        --final-visibility-min-views "${GAUSSIAN_FINAL_VISIBILITY_MIN_VIEWS}"
        --save-every "${GAUSSIAN_SAVE_EVERY}"
        --render-every "${GAUSSIAN_RENDER_EVERY}"
        --output-dir "${GAUSSIAN_OUTPUT_DIR}"
    )
    if [[ "${GAUSSIAN_DEPTH_LOSS}" == "1" ]]; then
        GSPLAT_CMD+=(--depth-loss)
    else
        GSPLAT_CMD+=(--no-depth-loss)
    fi
    if [[ "${GAUSSIAN_MASK_LOSS}" == "1" ]]; then
        GSPLAT_CMD+=(--mask-loss)
        if [[ -d "${GAUSSIAN_MASK_DIR}" ]]; then
            GSPLAT_CMD+=(--mask-dir "${GAUSSIAN_MASK_DIR}")
        fi
    else
        GSPLAT_CMD+=(--no-mask-loss)
    fi
    if [[ "${GAUSSIAN_DISTRIBUTED}" == "1" ]]; then
        GSPLAT_CMD+=(--distributed)
    fi
    run_step_with_cvd "${GAUSSIAN_CVD}" "${GSPLAT_CMD[@]}"
else
    log "skipping Gaussian training"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
    log "starting evaluation summary"
    EVAL_CMD=("${PYTHON_BIN}" evaluate.py --scene "${SCENE_DIR}" --output "${EVAL_REPORT_PATH}")
    if [[ -n "${RENDER_DIR_VGGT}" ]]; then
        require_path "${RENDER_DIR_VGGT}"
        EVAL_CMD+=(--render-dir-vggt "${RENDER_DIR_VGGT}")
    fi
    if [[ -n "${RENDER_DIR_BA}" ]]; then
        require_path "${RENDER_DIR_BA}"
        EVAL_CMD+=(--render-dir-ba "${RENDER_DIR_BA}")
    elif [[ "${GAUSSIAN_MODE}" == "ba" && -d "${GAUSSIAN_OUTPUT_DIR}/renders" ]]; then
        EVAL_CMD+=(--render-dir-ba "${GAUSSIAN_OUTPUT_DIR}/renders")
    fi
    if [[ -z "${RENDER_DIR_VGGT}" && "${GAUSSIAN_MODE}" == "vggt" && -d "${GAUSSIAN_OUTPUT_DIR}/renders" ]]; then
        EVAL_CMD+=(--render-dir-vggt "${GAUSSIAN_OUTPUT_DIR}/renders")
    fi
    run_step "${EVAL_CMD[@]}"
else
    log "skipping evaluation summary"
fi

if [[ "${RUN_VIEWER}" == "1" ]]; then
    log "starting viewer command"
    VIEWER_CMD=("${PYTHON_BIN}" viewer.py \
        --scene "${SCENE_DIR}" \
        --mode "${GAUSSIAN_MODE}" \
        --device "${DEVICE}" \
        --port "${VIEWER_PORT}")
    if [[ -f "${GAUSSIAN_OUTPUT_DIR}/checkpoint.pt" ]]; then
        VIEWER_CMD+=(--checkpoint "${GAUSSIAN_OUTPUT_DIR}/checkpoint.pt")
    fi
    run_step "${VIEWER_CMD[@]}"
fi

log "pipeline finished"
log "outputs: ${SCENE_DIR}"
log "run outputs: ${RUN_OUTPUT_DIR}"
