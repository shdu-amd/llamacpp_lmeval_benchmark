#!/bin/bash
# Launch a llama.cpp server for one gguf model and keep it running.
#
# This owns everything llama.cpp-specific about *serving* a model: the binary,
# context size, GPU offload, host/port, single-slot (serial) mode. The eval
# side (run_model_bench.py) never sees any of this — it just talks HTTP to the
# base URL this prints.
#
# Reads defaults from the environment (see active_environment.sh):
#   LLAMACPP_BUILD_DIR  -> $LLAMACPP_BUILD_DIR/bin/llama-server
#   LLAMACPP_HOST       -> bind host (default 127.0.0.1)
#   LLAMACPP_PORT       -> bind port (default 8080)
#
# Usage:
#   source scripts/active_environment.sh
#   ./scripts/launch_llama_server.sh --model models/Qwen3.5-0.8B-Q4_1.gguf
#   ./scripts/launch_llama_server.sh --model models/foo.gguf --ctx-size 8192 --n-gpu-layers 0
#
# Then, in another shell, point the eval at it:
#   python run_model_bench.py --base-url http://127.0.0.1:8080 \
#       --model-name Qwen3.5-0.8B --benchmark mmlu_pro --limit 100

set -euo pipefail

MODEL=""
CTX_SIZE=16384
N_GPU_LAYERS=-1          # -1 = offload all layers to GPU if possible
HOST="${LLAMACPP_HOST:-127.0.0.1}"
PORT="${LLAMACPP_PORT:-8080}"
SERVER_BIN="${LLAMACPP_SERVER_BIN:-}"

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model)         MODEL="$2"; shift 2 ;;
        --ctx-size)      CTX_SIZE="$2"; shift 2 ;;
        --n-gpu-layers)  N_GPU_LAYERS="$2"; shift 2 ;;
        --host)          HOST="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        --server-bin)    SERVER_BIN="$2"; shift 2 ;;
        -h|--help)       usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "error: --model <path-to-gguf> is required" >&2
    exit 2
fi
if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    exit 2
fi

# Resolve the server binary: explicit --server-bin, else from the build dir.
if [ -z "$SERVER_BIN" ]; then
    if [ -n "${LLAMACPP_BUILD_DIR:-}" ] && [ -x "$LLAMACPP_BUILD_DIR/bin/llama-server" ]; then
        SERVER_BIN="$LLAMACPP_BUILD_DIR/bin/llama-server"
    else
        echo "error: could not locate llama-server. Set \$LLAMACPP_BUILD_DIR" >&2
        echo "       (source active_environment.sh) or pass --server-bin." >&2
        exit 2
    fi
fi

echo "[serve] binary:        $SERVER_BIN"
echo "[serve] model:         $MODEL"
echo "[serve] ctx-size:      $CTX_SIZE"
echo "[serve] n-gpu-layers:  $N_GPU_LAYERS"
echo "[serve] listening on:  http://$HOST:$PORT"
echo "[serve] base-url ->    http://$HOST:$PORT   (pass this to run_model_bench.py --base-url)"

# --parallel N: number of concurrent request slots. --ctx-size is divided across
# them, so per-request context is --ctx-size / N. This must be >= the eval's
# --num-concurrent (run_model_bench.py) or extra in-flight requests queue.
# Runs in the foreground; Ctrl-C to stop.
exec "$SERVER_BIN" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --parallel 2
