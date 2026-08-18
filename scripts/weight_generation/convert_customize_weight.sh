#!/bin/bash
set -euo pipefail

# Usage: ./convert_customize_weight.sh <input-gguf> <imatrix-file> <output-gguf> [extra args...]
#   <input-gguf>    Path to the input GGUF file (BF16/F16/F32)
#   <imatrix-file>  Path to the importance matrix file
#   <output-gguf>   Path for the output quantized GGUF file
# Any extra args are passed through to the python script, e.g.:
#   --body-tensor-type q4_0   Body quant type (default: q4_1)
#   --pure                    Disable llama.cpp quant mixtures (forces exact body type,
#                             e.g. prevents q4_0 -> q4_1 on early ffn_down layers)
#   --tensor-type REGEX=TYPE  Override body tensors matching REGEX to TYPE, leaving the
#                             rest at --body-tensor-type. Repeatable; first match wins;
#                             applies even with --pure. e.g. --tensor-type attn_v=q8_0
#   --duplicate-tensor REGEX  Keep an extra original-dtype copy of every source tensor
#                             matching REGEX, named with '_duplicate' appended, alongside
#                             the quantized version. Repeatable. e.g. --duplicate-tensor attn_v

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd )

INPUT_GGUF="${1:-}"
IMATRIX="${2:-}"
OUTPUT="${3:-}"

if [ -z "$INPUT_GGUF" ] || [ -z "$IMATRIX" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input-gguf> <imatrix-file> <output-gguf> [extra args...]" >&2
    exit 1
fi

shift 3

python3 "$REPO_ROOT/quantize_to_desired_gguf.py" "$INPUT_GGUF" "$IMATRIX" "$OUTPUT" "$@"
