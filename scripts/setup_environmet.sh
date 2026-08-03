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
LMEVAL_SRC="$REPO_ROOT/lm-evaluation-harness"
LMMSEVAL_SRC="$REPO_ROOT/lmms-eval"
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

# 2. install lm-eval editable from source
echo "[setup] Installing lm-evaluation-harness (editable) from $LMEVAL_SRC"
#pip install -e "$LMEVAL_SRC"
pip install -e "$LMEVAL_SRC[api]"


# 2. install lmms-eval editable from source
# Reassert the CPU torch/torchvision pair afterwards: lmms-eval[all] can pull a
# PyPI torchvision that breaks the ABI match, so pin them from the CPU index last.
echo "[setup] Installing lmms-evaluation-harness (editable) from $LMMSEVAL_SRC"
#pip install -e "$LMEVAL_SRC"
pip install -e "$LMMSEVAL_SRC[all]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo "[setup] Done. Activate with:  source active_environment.sh"
