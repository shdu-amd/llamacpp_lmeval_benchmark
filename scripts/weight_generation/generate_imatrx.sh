#!/bin/bash

# Usage: ./generate_imatrx.sh <input-model> <calibration-data.txt> <imatrix-output>
MODEL="$1"
CALIBRATION_DATA="$2"
OUTPUT="$3"

if [ -z "$MODEL" ] || [ -z "$CALIBRATION_DATA" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input-model> <calibration-data.txt> <imatrix-output>" >&2
    exit 1
fi


# inpugt
# -m input model
# -f calibtration-data.txt

# -o imatrix output data  

"$LLAMACPP_BUILD_DIR/bin/llama-imatrix" -m "$MODEL" -f "$CALIBRATION_DATA"  -o "$OUTPUT"

# To view statistics for the generated imatrix afterwards, run:
#   "$LLAMACPP_BUILD_DIR/bin/llama-imatrix" --in-file "$OUTPUT" --show-statistics



