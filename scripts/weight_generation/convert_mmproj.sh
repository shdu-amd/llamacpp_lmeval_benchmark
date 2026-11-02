#!/bin/bash
set -euo pipefail

# Extract the vision part (multimodal projector, "mmproj") of a HuggingFace
# model into a dedicated GGUF, using llama.cpp's convert_hf_to_gguf.py --mmproj.
#
# Usage: ./convert_mmproj.sh <hf-model-dir> <output-mmproj-gguf> [outtype]
#   <hf-model-dir>       HuggingFace model directory (safetensors + config.json)
#   <output-mmproj-gguf> Path for the output mmproj GGUF
#   [outtype]            f32 | f16 | bf16 (default: bf16)
# Any further args are passed through to convert_hf_to_gguf.py.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd )
LLAMACPP_SRC="${LLAMACPP_SRC:-$REPO_ROOT/llama.cpp}"

HF_MODEL_DIR="${1:-}"
OUTPUT="${2:-}"
OUTTYPE="${3:-bf16}"

if [ -z "$HF_MODEL_DIR" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <hf-model-dir> <output-mmproj-gguf> [outtype=bf16] [extra args...]" >&2
    exit 1
fi

shift $(( $# >= 3 ? 3 : $# ))

# PYTHONSAFEPATH hides llama.cpp's own `conversion/` package, so turn it off
# for this single process (same trick as llamacpp_convert in active_environment.sh).
env -u PYTHONSAFEPATH python3 "$LLAMACPP_SRC/convert_hf_to_gguf.py" "$HF_MODEL_DIR" \
    --mmproj --outtype "$OUTTYPE" --outfile "$OUTPUT" "$@"
