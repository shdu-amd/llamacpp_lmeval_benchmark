#!/bin/bash
# Build llama.cpp with ROCm/HIP GPU acceleration (AMD gfx1151).
#
# Reads LLAMACPP_SRC and LLAMACPP_BUILD_DIR from the environment
# (see active_environment.sh). Falls back to sensible defaults.
#
# Usage:  source scripts/active_environment.sh && ./scripts/build_llamacpp_rocm.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
LLAMACPP_SRC="${LLAMACPP_SRC:-$REPO_ROOT/llama.cpp}"
LLAMACPP_BUILD_DIR="${LLAMACPP_BUILD_DIR:-$LLAMACPP_SRC/build}"

# Number of parallel build jobs
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "[build] LLAMACPP_SRC=$LLAMACPP_SRC"
echo "[build] LLAMACPP_BUILD_DIR=$LLAMACPP_BUILD_DIR"
echo "[build] JOBS=$JOBS"

# Resolve the HIP/ROCm toolchain. GGML_HIP=ON needs the ROCm clang as the
# HIP compiler; without HIPCXX/HIP_PATH the configure step fails to find HIP
# (or silently falls back to a CPU-only build).
if ! command -v hipconfig >/dev/null 2>&1; then
    echo "[build] ERROR: hipconfig not found — is ROCm installed and on PATH?" >&2
    exit 1
fi
HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}"
HIP_PATH="${HIP_PATH:-$(hipconfig -R)}"
GPU_TARGETS="${GPU_TARGETS:-gfx1151}"
export HIPCXX HIP_PATH
echo "[build] HIPCXX=$HIPCXX"
echo "[build] HIP_PATH=$HIP_PATH"
echo "[build] GPU_TARGETS=$GPU_TARGETS"

# Configure: ROCm/HIP GPU build, with the HTTP server enabled (needed by
# lm-eval's gguf backend, which talks to the server over HTTP).
cmake -S "$LLAMACPP_SRC" -B "$LLAMACPP_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="$GPU_TARGETS" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=OFF

# Build
cmake --build "$LLAMACPP_BUILD_DIR" --config Release -j "$JOBS"

echo "[build] Done. Binaries in $LLAMACPP_BUILD_DIR/bin"
