# Source this file to load the environment:  source scripts/active_environment.sh

# Resolve the repo root (parent of the scripts/ directory containing this script)
_THIS="${BASH_SOURCE[0]:-$0}"
export REPO_ROOT="$(cd "$(dirname "$_THIS")/.." && pwd)"

# llama.cpp source + build locations
export LLAMACPP_SRC="$REPO_ROOT/llama.cpp"
export LLAMACPP_BUILD_DIR="$LLAMACPP_SRC/build"

# Python virtual environment (created by setup_environmet.sh)
export VENV_DIR="$REPO_ROOT/.venv"

# llama.cpp server defaults (used by the gguf backend of lm-eval)
export LLAMACPP_HOST="${LLAMACPP_HOST:-127.0.0.1}"
export LLAMACPP_PORT="${LLAMACPP_PORT:-8080}"
export LLAMACPP_BASE_URL="http://${LLAMACPP_HOST}:${LLAMACPP_PORT}"

# Activate the venv if it exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "[active_environment] venv not found at $VENV_DIR — run setup_environmet.sh first" >&2
fi

# The evalscope submodule checkout at $REPO_ROOT/evalscope shares its name with
# the installed package (real code is one level down in evalscope/evalscope/).
# Launching Python from the repo root puts cwd on sys.path[0] and imports that
# outer folder as an empty namespace package, shadowing the editable install
# ("cannot import name 'TaskConfig' from 'evalscope'"). Safe-path drops
# cwd/script-dir from sys.path so imports resolve to the venv install.
export PYTHONSAFEPATH=1

echo "[active_environment] REPO_ROOT=$REPO_ROOT"
echo "[active_environment] LLAMACPP_SRC=$LLAMACPP_SRC"
echo "[active_environment] LLAMACPP_BUILD_DIR=$LLAMACPP_BUILD_DIR"
echo "[active_environment] LLAMACPP_BASE_URL=$LLAMACPP_BASE_URL"
