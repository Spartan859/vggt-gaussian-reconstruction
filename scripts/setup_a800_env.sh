#!/usr/bin/env bash
set -euo pipefail

# Build the A800 runtime environment for VGGT + gsplat main.
# Run this on the GPU machine, from this repository root.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-VGGT_GSPLAT_A800}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/share/micromamba/root/envs/${ENV_NAME}}"
MICROMAMBA="${MICROMAMBA:-/mnt/share/micromamba/bin/micromamba}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
CUDA_HOME="${CUDA_HOME:-${ENV_PREFIX}}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
CUDA_VERSION="${CUDA_VERSION:-11.8}"
TORCH_CUDA="${TORCH_CUDA:-cu118}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA}}"
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.1}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.pip_cache}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR}/torch_extensions_${TORCH_VERSION}_${TORCH_CUDA}}"
GSPLAT_INSTALL_MODE="${GSPLAT_INSTALL_MODE:-wheel}"
GSPLAT_WHEEL_VERSION="${GSPLAT_WHEEL_VERSION:-1.5.3+pt23cu118}"
GSPLAT_WHEEL_INDEX="${GSPLAT_WHEEL_INDEX:-https://docs.gsplat.studio/whl/pt23cu118/gsplat/}"
VGGT_PACKAGE_SPEC="${VGGT_PACKAGE_SPEC:-git+https://gh-proxy.org/https://github.com/facebookresearch/vggt.git}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
VGGT_WEIGHTS_PATH="${VGGT_WEIGHTS_PATH:-${TORCH_HOME}/hub/checkpoints/model.pt}"
VGGT_WEIGHTS_URL="${VGGT_WEIGHTS_URL:-https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt}"
VGGSFM_TRACKER_WEIGHTS_PATH="${VGGSFM_TRACKER_WEIGHTS_PATH:-${TORCH_HOME}/hub/checkpoints/vggsfm_v2_tracker.pt}"
VGGSFM_TRACKER_WEIGHTS_URL="${VGGSFM_TRACKER_WEIGHTS_URL:-https://hf-mirror.com/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt}"
WEIGHTS_DOWNLOAD_WORKERS="${WEIGHTS_DOWNLOAD_WORKERS:-8}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"

RUN_CREATE_ENV="${RUN_CREATE_ENV:-1}"
RUN_CUDA="${RUN_CUDA:-0}"
RUN_TORCH="${RUN_TORCH:-1}"
RUN_DEPS="${RUN_DEPS:-1}"
RUN_GSPLAT="${RUN_GSPLAT:-1}"
RUN_WEIGHTS="${RUN_WEIGHTS:-1}"
RUN_VERIFY="${RUN_VERIFY:-1}"
CLEAN_GSPLAT="${CLEAN_GSPLAT:-1}"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_a800_env.sh [options]

Options:
  --resume-from STAGE   Skip stages before STAGE. Stages: create-env, cuda, torch, deps, gsplat, weights, verify
  --skip-create-env     Do not create the micromamba environment
  --with-cuda-toolkit   Install CUDA toolkit packages before Python packages
  --skip-cuda           Do not install CUDA toolkit packages
  --skip-torch          Do not reinstall PyTorch CUDA wheels
  --skip-deps           Do not reinstall project/VGGT/example dependencies
  --skip-gsplat         Do not install/rebuild gsplat
  --skip-weights        Do not download VGGT weights
  --skip-clean-gsplat   Do not remove old gsplat build artifacts before installing/rebuilding
  --skip-verify         Do not run import/CUDA verification
  -h, --help            Show this help

Examples:
  bash scripts/setup_a800_env.sh
  bash scripts/setup_a800_env.sh --resume-from torch
  RUN_CUDA=1 GSPLAT_INSTALL_MODE=source bash scripts/setup_a800_env.sh --resume-from gsplat
  RUN_VERIFY=0 bash scripts/setup_a800_env.sh --resume-from deps
EOF
}

resume_from() {
    case "$1" in
        create-env)
            ;;
        cuda)
            RUN_CREATE_ENV=0
            ;;
        torch)
            RUN_CREATE_ENV=0
            RUN_CUDA=0
            ;;
        deps)
            RUN_CREATE_ENV=0
            RUN_CUDA=0
            RUN_TORCH=0
            ;;
        gsplat)
            RUN_CREATE_ENV=0
            RUN_CUDA=0
            RUN_TORCH=0
            RUN_DEPS=0
            ;;
        weights)
            RUN_CREATE_ENV=0
            RUN_CUDA=0
            RUN_TORCH=0
            RUN_DEPS=0
            RUN_GSPLAT=0
            ;;
        verify)
            RUN_CREATE_ENV=0
            RUN_CUDA=0
            RUN_TORCH=0
            RUN_DEPS=0
            RUN_GSPLAT=0
            RUN_WEIGHTS=0
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
        --skip-create-env)
            RUN_CREATE_ENV=0
            shift
            ;;
        --skip-cuda)
            RUN_CUDA=0
            shift
            ;;
        --with-cuda-toolkit)
            RUN_CUDA=1
            shift
            ;;
        --skip-torch)
            RUN_TORCH=0
            shift
            ;;
        --skip-deps)
            RUN_DEPS=0
            shift
            ;;
        --skip-gsplat)
            RUN_GSPLAT=0
            shift
            ;;
        --skip-weights)
            RUN_WEIGHTS=0
            shift
            ;;
        --skip-clean-gsplat)
            CLEAN_GSPLAT=0
            shift
            ;;
        --skip-verify)
            RUN_VERIFY=0
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

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"
cd "${REPO_ROOT}"

log() {
    echo "[$(date '+%F %T')] $*"
}

repair_ffmpeg_openh264() {
    if command -v ffprobe >/dev/null 2>&1 && ffprobe -version >/dev/null 2>&1; then
        return
    fi
    if [[ -e "${ENV_PREFIX}/lib/libopenh264.so.5" ]]; then
        return
    fi
    local target=""
    if [[ -e "${ENV_PREFIX}/lib/libopenh264.so.2.1.1" ]]; then
        target="libopenh264.so.2.1.1"
    elif [[ -e "${ENV_PREFIX}/lib/libopenh264.so" ]]; then
        target="libopenh264.so"
    fi
    if [[ -n "${target}" ]]; then
        log "repairing ffmpeg openh264 soname: libopenh264.so.5 -> ${target}"
        ln -s "${target}" "${ENV_PREFIX}/lib/libopenh264.so.5"
    fi
}

download_vggt_weights() {
    download_weight_file "VGGT weights" "${VGGT_WEIGHTS_URL}" "${VGGT_WEIGHTS_PATH}"
}

download_vggsfm_tracker_weights() {
    download_weight_file "VGGSfM tracker weights" "${VGGSFM_TRACKER_WEIGHTS_URL}" "${VGGSFM_TRACKER_WEIGHTS_PATH}"
}

download_weight_file() {
    local label="$1"
    local url="$2"
    local target_path="$3"
    mkdir -p "$(dirname "${target_path}")" "${HF_HOME}"
    if [[ -s "${target_path}" ]]; then
        log "${label} already exist: ${target_path}"
        return
    fi
    log "downloading ${label}"
    log "url: ${url}"
    log "path: ${target_path}"
    log "workers: ${WEIGHTS_DOWNLOAD_WORKERS}"
    local tmp_path="${target_path}.part"
    local downloaded=0
    if [[ -x "${PYTHON_BIN}" && "${WEIGHTS_DOWNLOAD_WORKERS}" -gt 1 ]]; then
        set +e
        WEIGHTS_URL="${url}" \
        WEIGHTS_PATH="${target_path}" \
        WEIGHTS_DOWNLOAD_WORKERS="${WEIGHTS_DOWNLOAD_WORKERS}" \
        "${PYTHON_BIN}" - <<'PY'
import concurrent.futures
import math
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request

url = os.environ["WEIGHTS_URL"]
target = Path(os.environ["WEIGHTS_PATH"])
workers = max(1, int(os.environ.get("WEIGHTS_DOWNLOAD_WORKERS", "8")))
tmp_path = target.with_name(target.name + ".part")
parts_dir = target.with_name(target.name + ".parts")

request = urllib.request.Request(url, method="HEAD")
with urllib.request.urlopen(request, timeout=60) as response:
    total = int(response.headers.get("Content-Length", "0"))
    ranges = response.headers.get("Accept-Ranges", "").lower()

if total <= 0 or "bytes" not in ranges:
    sys.exit(2)

parts_dir.mkdir(parents=True, exist_ok=True)
chunk_count = min(workers * 4, math.ceil(total / (64 * 1024 * 1024)))
chunk_size = math.ceil(total / chunk_count)
ranges_to_fetch = []
for idx in range(chunk_count):
    start = idx * chunk_size
    end = min(total - 1, start + chunk_size - 1)
    part = parts_dir / f"{idx:04d}.part"
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        continue
    ranges_to_fetch.append((idx, start, end, part, expected))

def fetch(item):
    idx, start, end, part, expected = item
    part_tmp = part.with_suffix(".part.tmp")
    for attempt in range(1, 6):
        try:
            existing = part_tmp.stat().st_size if part_tmp.exists() else 0
            if existing > expected:
                part_tmp.unlink()
                existing = 0
            if existing == expected:
                part_tmp.replace(part)
                return expected
            resume_start = start + existing
            headers = {"Range": f"bytes={resume_start}-{end}"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response, part_tmp.open("ab") as fh:
                shutil.copyfileobj(response, fh, length=1024 * 1024)
            if part_tmp.stat().st_size != expected:
                raise RuntimeError(f"chunk {idx} size mismatch: {part_tmp.stat().st_size} != {expected}")
            part_tmp.replace(part)
            return expected
        except Exception:
            if attempt == 5:
                raise
            time.sleep(3 * attempt)
    return 0

if ranges_to_fetch:
    done_bytes = total - sum(item[4] for item in ranges_to_fetch)
    print(f"parallel download: {chunk_count} chunks, {workers} workers, {done_bytes}/{total} bytes already complete", flush=True)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in ranges_to_fetch]
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            current = done_bytes + completed
            pct = current * 100 / total
            print(f"downloaded {current}/{total} bytes ({pct:.1f}%)", flush=True)
else:
    print(f"all {chunk_count} chunks already downloaded; assembling", flush=True)

with tmp_path.open("wb") as out:
    for idx in range(chunk_count):
        part = parts_dir / f"{idx:04d}.part"
        if not part.exists():
            raise FileNotFoundError(part)
        with part.open("rb") as fh:
            shutil.copyfileobj(fh, out, length=1024 * 1024)

if tmp_path.stat().st_size != total:
    raise RuntimeError(f"assembled size mismatch: {tmp_path.stat().st_size} != {total}")
shutil.rmtree(parts_dir)
PY
        status=$?
        set -e
        if [[ "${status}" == "0" ]]; then
            downloaded=1
        elif [[ "${status}" == "2" ]]; then
            echo "parallel range download is unavailable; falling back to curl/wget" >&2
        else
            echo "parallel range download failed with status ${status}; falling back to curl/wget" >&2
        fi
    fi
    if [[ "${downloaded}" != "1" ]]; then
        if command -v curl >/dev/null 2>&1; then
            curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 \
                --continue-at - \
                --output "${tmp_path}" \
                "${url}"
        elif command -v wget >/dev/null 2>&1; then
            wget --tries=5 --timeout=30 --continue \
                --output-document="${tmp_path}" \
                "${url}"
        else
            echo "Neither curl nor wget is available for downloading ${label}." >&2
            exit 1
        fi
    fi
    if [[ ! -s "${tmp_path}" ]]; then
        echo "Downloaded ${label} file is empty: ${tmp_path}" >&2
        exit 1
    fi
    mv "${tmp_path}" "${target_path}"
    log "downloaded ${label}: $(du -h "${target_path}" | awk '{print $1}')"
}

log "repo: ${REPO_ROOT}"
log "env: ${ENV_PREFIX}"
log "cuda toolkit: ${CUDA_VERSION}"
log "torch: ${TORCH_VERSION} torchvision: ${TORCHVISION_VERSION} index: ${TORCH_INDEX_URL}"
log "gsplat install: ${GSPLAT_INSTALL_MODE} ${GSPLAT_WHEEL_VERSION}"
log "vggt weights: ${VGGT_WEIGHTS_PATH}"
log "vggt weights url: ${VGGT_WEIGHTS_URL}"
log "vggsfm tracker weights: ${VGGSFM_TRACKER_WEIGHTS_PATH}"
log "vggsfm tracker weights url: ${VGGSFM_TRACKER_WEIGHTS_URL}"
log "weights download workers: ${WEIGHTS_DOWNLOAD_WORKERS}"
log "cuda arch: ${TORCH_CUDA_ARCH_LIST}"
log "stages: create-env=${RUN_CREATE_ENV} cuda=${RUN_CUDA} torch=${RUN_TORCH} deps=${RUN_DEPS} gsplat=${RUN_GSPLAT} weights=${RUN_WEIGHTS} clean-gsplat=${CLEAN_GSPLAT} verify=${RUN_VERIFY}"

log "configuring GitHub proxy rewrite for pip git dependencies"
if ! git config --global url."https://gh-proxy.org/https://github.com/".insteadOf "https://github.com/"; then
    log "warning: failed to write global git proxy config; continuing"
fi
if ! git config --global url."https://gh-proxy.org/https://github.com/".insteadOf "git@github.com:"; then
    log "warning: failed to write global git proxy config; continuing"
fi

if [[ "${RUN_CREATE_ENV}" == "1" && ! -x "${PYTHON_BIN}" ]]; then
    log "creating micromamba env"
    "${MICROMAMBA}" create -y -p "${ENV_PREFIX}" python=3.10 pip -c conda-forge
elif [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing env python: ${PYTHON_BIN}" >&2
    echo "Run without --skip-create-env, or choose an existing ENV_PREFIX." >&2
    exit 1
else
    log "skipping micromamba env creation"
fi

export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
export CUDA_HOME
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR
export TORCH_HOME HF_HOME VGGT_WEIGHTS_PATH VGGT_WEIGHTS_URL VGGSFM_TRACKER_WEIGHTS_PATH VGGSFM_TRACKER_WEIGHTS_URL

if [[ "${RUN_CUDA}" == "1" ]]; then
    log "installing CUDA ${CUDA_VERSION} toolkit and ninja"
    "${MICROMAMBA}" install -y -p "${ENV_PREFIX}" \
        cuda-toolkit="${CUDA_VERSION}" ninja \
        -c nvidia -c conda-forge
else
    log "skipping CUDA package install"
fi

if [[ "${RUN_TORCH}" == "1" ]]; then
    log "installing torch ${TORCH_VERSION} ${TORCH_CUDA} wheels"
    TMPDIR="${TMPDIR}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${PYTHON_BIN}" -m pip install --force-reinstall \
        torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"
else
    log "skipping torch install"
fi

if [[ "${RUN_DEPS}" == "1" ]]; then
    FILTERED_REQ_DIR="${TMPDIR}/filtered_requirements"
    mkdir -p "${FILTERED_REQ_DIR}"
    cat > "${FILTERED_REQ_DIR}/vggt_demo_deps.txt" <<'EOF'
gradio==5.17.1
viser==0.2.23
tqdm
hydra-core
omegaconf
opencv-python
scipy
onnxruntime
requests
trimesh
matplotlib
pydantic==2.10.6
pycolmap==3.10.0
pyceres==2.3
EOF

    log "installing build helpers and numpy ${NUMPY_VERSION}"
    TMPDIR="${TMPDIR}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${PYTHON_BIN}" -m pip install --upgrade \
        pip setuptools wheel packaging ninja jaxtyping nvtx "rich>=12"
    TMPDIR="${TMPDIR}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${PYTHON_BIN}" -m pip install --force-reinstall \
        numpy=="${NUMPY_VERSION}"

    log "installing project"
    TMPDIR="${TMPDIR}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${PYTHON_BIN}" -m pip install -e .

    log "installing VGGT package"
    TMPDIR="${TMPDIR}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${PYTHON_BIN}" -m pip install "${VGGT_PACKAGE_SPEC}" --no-deps

    log "installing VGGT runtime dependencies"
    TMPDIR="${TMPDIR}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${PYTHON_BIN}" -m pip install \
        "Pillow" huggingface_hub einops safetensors \
        -r "${FILTERED_REQ_DIR}/vggt_demo_deps.txt"
    TMPDIR="${TMPDIR}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${PYTHON_BIN}" -m pip install \
        "git+https://gh-proxy.org/https://github.com/jytime/LightGlue.git#egg=lightglue"

    repair_ffmpeg_openh264
else
    log "skipping project/VGGT/example dependency install"
    repair_ffmpeg_openh264
fi

if [[ "${RUN_GSPLAT}" == "1" ]]; then
    if [[ "${CLEAN_GSPLAT}" == "1" ]]; then
        log "cleaning old gsplat build artifacts"
        if [[ -d external/gsplat ]]; then
            find external/gsplat \
                \( -name 'csrc*.so' -o -name '*.egg-info' -o -name 'build' \) \
                -print -exec rm -rf {} +
        fi
        rm -rf "${TORCH_EXTENSIONS_DIR}"
    fi
    if [[ "${GSPLAT_INSTALL_MODE}" == "wheel" ]]; then
        log "installing gsplat wheel ${GSPLAT_WHEEL_VERSION}"
        "${PYTHON_BIN}" -m pip uninstall -y gsplat || true
        TMPDIR="${TMPDIR}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${PYTHON_BIN}" -m pip install --force-reinstall \
            "gsplat==${GSPLAT_WHEEL_VERSION}" \
            --find-links "${GSPLAT_WHEEL_INDEX}" \
            --no-deps
    elif [[ "${GSPLAT_INSTALL_MODE}" == "source" ]]; then
        if [[ ! -d external/gsplat ]]; then
            echo "GSPLAT_INSTALL_MODE=source requires external/gsplat." >&2
            exit 1
        fi
        log "building gsplat main CUDA extensions from source"
        TMPDIR="${TMPDIR}" \
        PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
        CUDA_HOME="${CUDA_HOME}" \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        BUILD_EXPERIMENTAL="${BUILD_EXPERIMENTAL:-0}" \
        MAX_JOBS="${MAX_JOBS:-8}" \
        "${PYTHON_BIN}" -m pip install -e external/gsplat --no-build-isolation --force-reinstall
    else
        echo "Unknown GSPLAT_INSTALL_MODE: ${GSPLAT_INSTALL_MODE}" >&2
        echo "Expected: wheel or source" >&2
        exit 2
    fi
else
    log "skipping gsplat build"
fi

if [[ "${RUN_WEIGHTS}" == "1" ]]; then
    download_vggt_weights
    download_vggsfm_tracker_weights
else
    log "skipping VGGT/VGGSfM weights download"
fi

if [[ "${RUN_VERIFY}" == "1" ]]; then
    log "verifying imports"
    MPLCONFIGDIR="${TMPDIR}/matplotlib" \
    "${PYTHON_BIN}" - <<'PY'
import importlib
import os
from pathlib import Path
import torch

mods = [
    "vggt",
    "vggt.models.vggt",
    "pycolmap",
    "pyceres",
    "gsplat",
    "gsplat.rendering",
]
for name in mods:
    importlib.import_module(name)
    print("OK", name)
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available(), "gpu count", torch.cuda.device_count())
for env_name in ("VGGT_WEIGHTS_PATH", "VGGSFM_TRACKER_WEIGHTS_PATH"):
    path = Path(os.environ[env_name])
    print(env_name, path, path.exists(), path.stat().st_size if path.exists() else 0)
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Missing required weights: {path}")
if os.environ.get("REQUIRE_CUDA", "0") == "1" and not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; rerun on a GPU node or set REQUIRE_CUDA=0.")
PY
else
    log "skipping verification"
fi

log "done"
