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

# 1. create a new python venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# 3. install pytorch (CPU build — no GPU detected on this host)
echo "[setup] Installing PyTorch (CPU)"
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. install lm-eval editable from source
echo "[setup] Installing lm-evaluation-harness (editable) from $LMEVAL_SRC"
#pip install -e "$LMEVAL_SRC"
pip install -e "$LMEVAL_SRC[api]"
echo "[setup] Done. Activate with:  source active_environment.sh"
