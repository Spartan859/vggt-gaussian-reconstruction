#!/usr/bin/env bash
set -euo pipefail

# Compile and install external/gsplat on a GPU machine.
# This is intended for the VGGT_GSPLAT_A800 environment and A800 GPUs.
#
# Notes:
# - external/gsplat is expected to be checked out at tag v1.5.3 for the
#   torch 2.3 + CUDA 11.8 environment. That tag compiles with -std=c++17.
# - If the checked-out source requests -std=c++20, CUDA 11.8 nvcc cannot
#   compile it; use a CUDA 12.x toolkit or switch back to v1.5.3.
# - This script does not install external/gsplat/examples/requirements.txt,
#   because that file pins torch==2.9.1 and would replace the cu118 torch env.

REPO_ROOT="${REPO_ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/share/micromamba/root/envs/VGGT_GSPLAT_A800}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
GSPLAT_REPO="${GSPLAT_REPO:-${REPO_ROOT}/external/gsplat}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
MAX_JOBS="${MAX_JOBS:-8}"
FORCE="${FORCE:-0}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
SKIP_IF_IMPORT_OK="${SKIP_IF_IMPORT_OK:-1}"

log() {
    printf '[%(%F %T)T] %s\n' -1 "$*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_path() {
    [[ -e "$1" ]] || die "Missing required path: $1"
}

cuda_major() {
    local nvcc="${CUDA_HOME}/bin/nvcc"
    "${nvcc}" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1
}

gsplat_ref() {
    git -C "${GSPLAT_REPO}" describe --tags --always --dirty 2>/dev/null || echo "unknown"
}

gsplat_uses_cpp20() {
    grep -R -q -- "-std=c++20" \
        "${GSPLAT_REPO}/setup.py" \
        "${GSPLAT_REPO}/gsplat/cuda/build.py" 2>/dev/null
}

verify_compiled_gsplat() {
    PYTHONPATH="${GSPLAT_REPO}:${GSPLAT_REPO}/examples:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
import gsplat
print("gsplat", gsplat.__file__)
from gsplat import csrc
print("csrc", csrc.__file__)
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam
from gsplat.compression import PngCompression
print("compiled gsplat ok")
PY
}

verify_external_gsplat() {
    PYTHONPATH="${GSPLAT_REPO}:${GSPLAT_REPO}/examples:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
import gsplat
print("gsplat", gsplat.__file__)
from gsplat import csrc
print("csrc", csrc.__file__)
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam
from gsplat.compression import PngCompression
from fused_ssim import fused_ssim
from nerfview import CameraState, RenderTabState
import gsplat_viewer
print("rendering/strategy/optimizer/compression/fused_ssim/nerfview/viewer ok")
PY
}

require_path "${PYTHON_BIN}"
require_path "${GSPLAT_REPO}/setup.py"
require_path "${CUDA_HOME}/bin/nvcc"

GSPLAT_REF="$(gsplat_ref)"
log "repo: ${REPO_ROOT}"
log "env: ${ENV_PREFIX}"
log "python: ${PYTHON_BIN}"
log "gsplat repo: ${GSPLAT_REPO}"
log "gsplat ref: ${GSPLAT_REF}"
log "CUDA_HOME: ${CUDA_HOME}"
log "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
log "MAX_JOBS: ${MAX_JOBS}"
log "FORCE_REINSTALL: ${FORCE_REINSTALL}"
log "SKIP_IF_IMPORT_OK: ${SKIP_IF_IMPORT_OK}"

"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY

major="$(cuda_major)"
log "nvcc major: ${major:-unknown}"
if [[ -z "${major}" ]]; then
    die "Could not parse nvcc version from ${CUDA_HOME}/bin/nvcc --version"
fi
if (( major < 12 )) && gsplat_uses_cpp20 && [[ "${FORCE}" != "1" ]]; then
    cat >&2 <<EOF
ERROR: this external/gsplat checkout uses -std=c++20, but CUDA ${major}.x nvcc does not support it.

Use one of these:
  1. Switch external/gsplat to tag v1.5.3:
       git -C external/gsplat checkout v1.5.3
       git -C external/gsplat submodule update --init --recursive

  2. Load a CUDA 12.x toolkit on the GPU machine, e.g.
       export CUDA_HOME=/usr/local/cuda-12.1
       export PATH=\$CUDA_HOME/bin:\$PATH
       export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
       bash scripts/compile_external_gsplat_gpu.sh

  3. Keep the prebuilt gsplat wheel for torch 2.3 + cu118 instead of compiling external/gsplat.

Set FORCE=1 only if you have patched external/gsplat to compile with your nvcc.
EOF
    exit 2
fi
if (( major < 12 )); then
    log "CUDA ${major}.x accepted because this gsplat checkout does not request -std=c++20"
fi

if [[ "${SKIP_IF_IMPORT_OK}" == "1" && "${FORCE_REINSTALL}" != "1" ]]; then
    log "checking whether external/gsplat is already compiled"
    if verify_external_gsplat; then
        log "external/gsplat already imports correctly; skipping rebuild"
        exit 0
    fi
    log "external/gsplat import failed; rebuilding"
fi

log "installing lightweight simple_trainer dependencies without changing torch"
"${PYTHON_BIN}" -m pip install \
    --no-deps \
    ninja tyro viser imageio opencv-python-headless tqdm torchmetrics tensorboard pyyaml matplotlib splines piexif tensorly || true
"${PYTHON_BIN}" -m pip install \
    --no-deps \
    "git+https://gh-proxy.org/https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787#egg=nerfview" \
    "git+https://gh-proxy.org/https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5#egg=fused-ssim" || true

if [[ "${FORCE_REINSTALL}" != "1" ]]; then
    log "checking whether compiled external/gsplat can be reused"
    if verify_compiled_gsplat; then
        log "compiled external/gsplat is reusable; skipping editable rebuild"
        log "verifying external/gsplat import"
        verify_external_gsplat
        log "done"
        exit 0
    fi
    log "compiled external/gsplat is not reusable; rebuilding"
fi

log "compiling and installing external/gsplat editable"
(
    cd "${REPO_ROOT}"
    pip_args=(-e "${GSPLAT_REPO}" --no-build-isolation --no-deps)
    if [[ "${FORCE_REINSTALL}" == "1" ]]; then
        pip_args+=(--force-reinstall)
    fi
    CUDA_HOME="${CUDA_HOME}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
    "${PYTHON_BIN}" -m pip install "${pip_args[@]}"
)

log "verifying external/gsplat import"
verify_external_gsplat

log "done"
