#!/bin/bash
# Build llama.cpp (CPU by default).
#
# Reads LLAMACPP_SRC and LLAMACPP_BUILD_DIR from the environment
# (see active_environment.sh). Falls back to sensible defaults.
#
# Usage:  source scripts/active_environment.sh && ./scripts/build_llamacpp.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
LLAMACPP_SRC="${LLAMACPP_SRC:-$REPO_ROOT/llama.cpp}"
LLAMACPP_BUILD_DIR="${LLAMACPP_BUILD_DIR:-$LLAMACPP_SRC/build}"

# Number of parallel build jobs
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "[build] LLAMACPP_SRC=$LLAMACPP_SRC"
echo "[build] LLAMACPP_BUILD_DIR=$LLAMACPP_BUILD_DIR"
echo "[build] JOBS=$JOBS"

# Configure: CPU-only build, with the HTTP server enabled (needed by
# lm-eval's gguf backend, which talks to the server over HTTP).
cmake -S "$LLAMACPP_SRC" -B "$LLAMACPP_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=OFF

# Build
cmake --build "$LLAMACPP_BUILD_DIR" --config Release -j "$JOBS"

echo "[build] Done. Binaries in $LLAMACPP_BUILD_DIR/bin"
