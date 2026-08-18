#!/bin/bash
# Build llama.cpp with NVIDIA CUDA GPU acceleration.
#
# Reads LLAMACPP_SRC and LLAMACPP_BUILD_DIR from the environment
# (see active_environment.sh). Falls back to sensible defaults.
#
# Usage:  source scripts/active_environment.sh && ./scripts/build_llamacpp_cuda.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
LLAMACPP_SRC="${LLAMACPP_SRC:-$REPO_ROOT/llama.cpp}"
LLAMACPP_BUILD_DIR="${LLAMACPP_BUILD_DIR:-$LLAMACPP_SRC/build}"

# Number of parallel build jobs
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "[build] LLAMACPP_SRC=$LLAMACPP_SRC"
echo "[build] LLAMACPP_BUILD_DIR=$LLAMACPP_BUILD_DIR"
echo "[build] JOBS=$JOBS"

# Resolve the CUDA toolkit. GGML_CUDA=ON needs nvcc; without it the configure
# step fails to find CUDA (or silently falls back to a CPU-only build).
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[build] ERROR: nvcc not found — is the CUDA toolkit installed and on PATH?" >&2
    echo "[build]        (add \$CUDA_HOME/bin to PATH, e.g. /usr/local/cuda/bin)" >&2
    exit 1
fi
echo "[build] nvcc=$(command -v nvcc)"
nvcc --version | sed -n 's/^/[build] /p'

# CUDA compute capabilities to target. Leave unset to let CMake/nvcc pick
# defaults, or set e.g. CMAKE_CUDA_ARCHITECTURES=89 for an RTX 4090.
CUDA_ARCH_ARG=()
if [ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
    CUDA_ARCH_ARG=(-DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES")
    echo "[build] CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
fi

# Configure: CUDA GPU build, with the HTTP server enabled (needed by
# lm-eval's gguf backend, which talks to the server over HTTP).
cmake -S "$LLAMACPP_SRC" -B "$LLAMACPP_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    "${CUDA_ARCH_ARG[@]}" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=OFF

# Build
cmake --build "$LLAMACPP_BUILD_DIR" --config Release -j "$JOBS"

echo "[build] Done. Binaries in $LLAMACPP_BUILD_DIR/bin"
