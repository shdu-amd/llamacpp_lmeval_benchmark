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
# For a multimodal model (VLM), also pass the projector so the server accepts
# image/audio inputs:
#   ./scripts/launch_llama_server.sh --model models/Qwen2-VL-7B-Q4.gguf \
#       --mmproj models/Qwen2-VL-7B-mmproj-f16.gguf
#
# By default the total --ctx-size is sized so each of the --parallel slots gets
# --ctx-per-slot tokens of KV cache (default 4 slots x 16384 = 65536). Override
# per-slot size or slot count directly:
#   ./scripts/launch_llama_server.sh --model models/foo.gguf --parallel 8 --ctx-per-slot 16384
# Passing --ctx-size explicitly overrides this (per-slot becomes ctx-size/parallel).
#
# Thinking is disabled by default (--reasoning off) so reasoning models put their
# answer in message.content, which lm-eval reads. With thinking on, a small model
# can ruminate past --max-tokens and never close </think>, returning empty content
# (scored [invalid]). Re-enable with: --reasoning on (or auto).
#
# Then, in another shell, point the eval at it:
#   python run_model_bench.py --base-url http://127.0.0.1:8080 \
#       --model-name Qwen3.5-0.8B --benchmark mmlu_pro --limit 100

set -euo pipefail

MODEL=""
MMPROJ=""                # optional multimodal projector (--mmproj). When set, the
                        # server loads a vision/audio projector alongside the model
                        # so it can accept image/audio inputs (VLMs like Qwen-VL,
                        # Gemma 3, etc.). Leave empty for text-only models.
PARALLEL=24               # number of concurrent request slots
CTX_PER_SLOT=8192       # per-slot KV cache; total ctx = CTX_PER_SLOT * PARALLEL
CTX_SIZE=""              # if set explicitly, overrides CTX_PER_SLOT * PARALLEL
N_GPU_LAYERS=-1          # -1 = offload all layers to GPU if possible
NO_MMAP=1               # 1 = pass --no-mmap. On this Strix Halo APU the BIOS carves
                        # most DRAM into the GPU pool, leaving a small CPU-visible
                        # RAM pool. Default mmap keeps the whole GGUF mapped there as
                        # page cache (double-counting the weights + swapping), so we
                        # disable it and load tensors straight into the VRAM pool.
                        # Set --mmap to re-enable on machines with plenty of host RAM.
REASONING="off"         # thinking mode: off|on|auto. off so reasoning models emit
                        # the answer in message.content (lm-eval reads content, not
                        # reasoning_content); with thinking on, small models can
                        # ruminate past the token limit and return empty content.
CACHE_RAM=0             # llama.cpp -cram/--cache-ram (MiB). The server saves idle
                        # slots' KV state into this RAM prompt cache; with many
                        # --parallel slots on unique benchmark prompts that cache
                        # only thrashes (save/evict/restore stalls of several
                        # seconds -> clients time out / disconnect). 0 disables it
                        # for near-zero-hit sweeps; -1 = no limit, >0 = MiB cap.
FLASH_ATTN="on"         # llama.cpp -fa/--flash-attn (on|off|auto). Shrinks the KV
                        # cache and cuts attention bandwidth, which directly eases
                        # the memory pressure from many --parallel slots on this
                        # APU. Default 'on'; set 'auto' to let llama.cpp decide, or
                        # 'off' to compare if a backend kernel is slower.
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
        --mmproj)        MMPROJ="$2"; shift 2 ;;
        --parallel)      PARALLEL="$2"; shift 2 ;;
        --ctx-per-slot)  CTX_PER_SLOT="$2"; shift 2 ;;
        --ctx-size)      CTX_SIZE="$2"; shift 2 ;;
        --n-gpu-layers)  N_GPU_LAYERS="$2"; shift 2 ;;
        --no-mmap)       NO_MMAP=1; shift ;;
        --mmap)          NO_MMAP=0; shift ;;
        --reasoning)     REASONING="$2"; shift 2 ;;
        --cache-ram)     CACHE_RAM="$2"; shift 2 ;;
        --flash-attn)    FLASH_ATTN="$2"; shift 2 ;;
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
if [ -n "$MMPROJ" ] && [ ! -f "$MMPROJ" ]; then
    echo "error: mmproj not found: $MMPROJ" >&2
    exit 2
fi

# Total ctx is split evenly across --parallel slots by llama.cpp, so to give each
# slot CTX_PER_SLOT tokens we request CTX_PER_SLOT * PARALLEL total. An explicit
# --ctx-size overrides this (per-slot then becomes CTX_SIZE / PARALLEL).
if [ -z "$CTX_SIZE" ]; then
    CTX_SIZE=$(( CTX_PER_SLOT * PARALLEL ))
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
echo "[serve] mmproj:        $( [ -n "$MMPROJ" ] && echo "$MMPROJ" || echo '(none, text-only)' )"
echo "[serve] parallel:      $PARALLEL"
echo "[serve] ctx-size:      $CTX_SIZE  (~$(( CTX_SIZE / PARALLEL )) per slot)"
echo "[serve] n-gpu-layers:  $N_GPU_LAYERS"
echo "[serve] mmap:          $( [ "$NO_MMAP" = 1 ] && echo 'off (--no-mmap)' || echo 'on' )"
echo "[serve] reasoning:     $REASONING"
echo "[serve] cache-ram:     $CACHE_RAM MiB$( [ "$CACHE_RAM" = 0 ] && echo '  (prompt cache disabled)' )"
echo "[serve] flash-attn:    $FLASH_ATTN"
echo "[serve] listening on:  http://$HOST:$PORT"
echo "[serve] base-url ->    http://$HOST:$PORT   (pass this to run_model_bench.py --base-url)"

# --parallel N: number of concurrent request slots. --ctx-size is divided across
# them, so per-request context is --ctx-size / N. This must be >= the eval's
# --num-concurrent (run_model_bench.py) or extra in-flight requests queue.
# Runs in the foreground; Ctrl-C to stop.
MMAP_ARGS=()
if [ "$NO_MMAP" = 1 ]; then
    MMAP_ARGS+=(--no-mmap)
fi
MMPROJ_ARGS=()
if [ -n "$MMPROJ" ]; then
    MMPROJ_ARGS+=(--mmproj "$MMPROJ")
fi
exec "$SERVER_BIN" \
    --model "$MODEL" \
    "${MMPROJ_ARGS[@]}" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --reasoning "$REASONING" \
    --cache-ram "$CACHE_RAM" \
    --flash-attn "$FLASH_ATTN" \
    "${MMAP_ARGS[@]}" \
    --parallel "$PARALLEL" 