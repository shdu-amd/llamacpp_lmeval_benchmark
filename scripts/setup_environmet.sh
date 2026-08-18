#!/bin/bash
# Set up the Python environment for lm-evaluation-harness.
#
# 1. create a new python venv
# 2. install lm-eval (editable) from source in ./lm-evaluation-harness
# 3. install pytorch
#
# Usage:  ./scripts/setup_environmet.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

EVALSCOPE_SRC="$REPO_ROOT/evalscope"
# 1. create a new python venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# 3. install pytorch (CPU build — no GPU detected on this host)
# torch and torchvision MUST come from the same CPU index in one command, or
# pip resolves torchvision from PyPI against a mismatched torch ABI and imports
# fail at runtime (RuntimeError: operator torchvision::nms does not exist).
echo "[setup] Installing PyTorch + torchvision (CPU)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. install evalscope editable from source, with every optional extra it
# defines. Extras are declared in evalscope/pyproject.toml under
# [tool.setuptools.dynamic.optional-dependencies] and the [project.optional-
# dependencies] file-mapping block. We install each extra individually so one
# unresolvable extra doesn't abort the whole environment; failures are logged
# and the script continues.
echo "[setup] Installing evalscope (base) from $EVALSCOPE_SRC"
pip install -e "$EVALSCOPE_SRC"

# All optional-dependency extras defined in pyproject.toml. Keep in sync with
# that file if new benchmarks/extras are added.
EVALSCOPE_EXTRAS=(
    api
    opencompass vlmeval rag perf app aigc sandbox service mcp dev docs
    ocr_bench omnidoc_bench maritime_ocr_bench miniwob ifeval ifbench
    multi_if olympiad_bench needle_haystack openai_mrcr general_arena
    arena_hard air_bench caption refcoco swe_bench wmt torgo terminal_bench
    deep_swe toolathlon bfcl wide_search
)

FAILED_EXTRAS=()
for extra in "${EVALSCOPE_EXTRAS[@]}"; do
    echo "[setup] Installing evalscope extra: $extra"
    if ! pip install -e "${EVALSCOPE_SRC}[${extra}]"; then
        echo "[setup] WARNING: failed to install extra '$extra' — continuing"
        FAILED_EXTRAS+=("$extra")
    fi
done

if [ "${#FAILED_EXTRAS[@]}" -gt 0 ]; then
    echo "[setup] The following extras failed to install: ${FAILED_EXTRAS[*]}"
    echo "[setup] Re-run manually to see the error, e.g.:"
    echo "[setup]   pip install -e '${EVALSCOPE_SRC}[${FAILED_EXTRAS[0]}]'"
fi

pip install flask
pip install fastapi
pip install uvicorn
pip install sse_starlette

# gguf Python package, required by scripts/weight_generation/*.
# Install the copy vendored in the llama.cpp tree so it stays in sync with the
# llama.cpp binaries this repo builds, rather than a mismatched PyPI release.
echo "[setup] Installing gguf (from llama.cpp/gguf-py)"
pip install -e "$REPO_ROOT/llama.cpp/gguf-py"

# 4. build the web dashboard (React/Vite -> evalscope/web/dist)
# The `evalscope service` dashboard is a compiled SPA served from web/dist.
# Without dist, Flask serves only the JSON API and every page URL 404s
# (the standalone reports/report.html still works without this build).
# Node.js + npm are required; install whichever version you prefer, e.g. via
# nvm (https://github.com/nvm-sh/nvm) or your distro. Node >=18 is recommended.
WEB_SRC="$EVALSCOPE_SRC/evalscope/web"

# nvm only auto-loads in interactive shells, so source it here to put node/npm
# on PATH (also picks up the default alias). npm ships with node — no separate
# source needed. Harmless if nvm isn't installed.
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh"
    nvm use default >/dev/null 2>&1 || true
fi

if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    echo "[setup] Building web dashboard with node $(node --version), npm $(npm --version)"
    (
        cd "$WEB_SRC"
        if [ -f package-lock.json ]; then
            npm ci
        else
            npm install
        fi
        npm run build
    )
    echo "[setup] Dashboard built. Launch: evalscope service --outputs results/"
    echo "[setup]   then open http://127.0.0.1:9000/dashboard"
else
    echo "[setup] WARNING: node/npm not found — skipping web dashboard build."
    echo "[setup]   Install Node.js + npm (>=18 recommended, e.g. via nvm), then run:"
    echo "[setup]     cd $WEB_SRC && npm ci && npm run build"
    echo "[setup]   Until then, view results via the standalone reports/report.html."
fi
