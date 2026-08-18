
# given an input imatrix, it will quantize the body (default q4_1, e.g. q4_0) except lm_head to q8_0, embedding to bf16,
# and any tensor whose name ends with ".bias" or "_norm.weight" to bf16.
#
# Two-pass approach:
#   Pass 1: llama-quantize on the original bf16 source (norm/bias tensors stay F32 after quantize)
#   Pass 2: Python gguf copy pass reads the quantized file, converts remaining F32 norm/bias → bf16
#
# Tied embeddings are handled adaptively based on the source GGUF's tensors:
#   - untied (a separate output.weight exists): lm_head and token_embd are quantized
#     independently via --output-tensor-type and --token-embedding-type.
#   - tied (only token_embd.weight exists): the single shared tensor is quantized to the
#     lower-precision of the two configured types, forced via --token-embedding-type (which
#     llama-quantize applies to token_embd before its tied logic runs).

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# llama.cpp lives under this repo; allow an env override for non-standard layouts.
REPO_ROOT = Path(__file__).resolve().parent
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", REPO_ROOT / "llama.cpp"))

GGUF_PY = LLAMA_CPP_DIR / "gguf-py"
sys.path.insert(0, str(GGUF_PY))
import gguf  # noqa: E402

LLAMA_CPP_BUILD_DIR = Path(os.environ.get("LLAMA_CPP_BUILD_DIR", LLAMA_CPP_DIR / "build"))
LLAMA_QUANTIZE_BIN = LLAMA_CPP_BUILD_DIR / "bin" / "llama-quantize"

# Tensors whose names match these patterns will be converted from f32 → bf16.
_F32_TO_BF16 = re.compile(r"(_norm\.weight|\.bias)$")


def run(cmd: list) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _bits_per_weight(quant_type: str) -> float:
    """Bits-per-weight of a GGUF quant type, from the gguf library's block/type sizes."""
    try:
        block_size, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[quant_type.upper()]]
    except KeyError as e:
        raise ValueError(f"Unknown GGUF quant type {quant_type!r}") from e
    return type_size * 8 / block_size


def lower_precision_type(a: str, b: str) -> str:
    """Return whichever of the two quant types has fewer bits-per-weight (ties -> a)."""
    return a if _bits_per_weight(a) <= _bits_per_weight(b) else b


def has_unique_lm_head(gguf_path: Path) -> bool:
    """True if the source GGUF carries a distinct output projection (untied embeddings),
    i.e. a separate 'output.weight' tensor rather than sharing token_embd.weight."""
    reader = gguf.GGUFReader(str(gguf_path), mode="r")
    for tensor in reader.tensors:
        if tensor.name == "output.weight" or "lm_head.weight" in tensor.name:
            return True
    return False


def copy_with_duplicates(
    quantized_gguf: Path,
    source_gguf: Path,
    output_gguf: Path,
    duplicate_patterns: list[str],
) -> None:
    """Copy the quantized GGUF to output, appending original-dtype duplicates of every
    source tensor whose name matches one of duplicate_patterns (regex_search).

    Each spec is either "REGEX" or "REGEX=REPLACEMENT":
      - "REGEX"             -> duplicate name is "<name>_duplicate"
      - "REGEX=REPLACEMENT" -> duplicate name is re.sub(REGEX, REPLACEMENT, name), so
                               backreferences (\\1, \\g<0>, ...) can build per-tensor names.

    Each duplicate keeps the source tensor's original dtype, for later debug/comparison
    against the quantized copy.
    """
    # Parse specs into (compiled_pattern, replacement-or-None).
    specs = []
    for spec in duplicate_patterns:
        regex, sep, replacement = spec.partition("=")
        specs.append((re.compile(regex), replacement if sep else None))

    reader = gguf.GGUFReader(str(quantized_gguf), mode="r")
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(str(output_gguf), arch, endianess=reader.endianess)

    # Copy all metadata fields using the library's own field.contents() decoder.
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), vtype, sub_type=sub_type)

    # Collect matching tensors from the original source (kept in their original dtype).
    src_reader = gguf.GGUFReader(str(source_gguf), mode="r")
    existing_names = {t.name for t in reader.tensors}
    duplicates = []  # (dup_name, source_tensor)
    seen_dup_names: set[str] = set()
    for t in src_reader.tensors:
        for pattern, replacement in specs:
            if not pattern.search(t.name):
                continue
            if replacement is not None:
                try:
                    dup_name = pattern.sub(replacement, t.name)
                except re.error as e:
                    raise ValueError(
                        f"Invalid --duplicate-tensor replacement {replacement!r}: {e}. "
                        f"The part after '=' is an re.sub template, NOT a regex — plain regex "
                        f"syntax like \\d or [..] is not valid there. Use backreferences (\\1, "
                        f"\\g<0>) to insert captured groups, e.g. "
                        f"'blk\\.(\\d+)\\.ssm_alpha\\.weight=blk.\\1.ssm_alpha.bf16.weight'."
                    ) from e
            else:
                dup_name = f"{t.name}_duplicate"
            if dup_name in existing_names or dup_name in seen_dup_names:
                raise ValueError(
                    f"Duplicate tensor name collision: {dup_name!r} (from source tensor "
                    f"{t.name!r}) already exists. Use a REPLACEMENT with backreferences to "
                    f"produce unique names, e.g. --duplicate-tensor 'REGEX=\\g<0>_duplicate'."
                )
            seen_dup_names.add(dup_name)
            duplicates.append((dup_name, t))
            break  # first matching spec wins

    # Register tensor info: all quantized tensors first, then the duplicates.
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
    for dup_name, t in duplicates:
        writer.add_tensor_info(dup_name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    print(f"  Duplicating {len(duplicates)} tensors (original dtype): "
          f"{[d[0] for d in duplicates][:5]}{'...' if len(duplicates) > 5 else ''}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Stream tensor data in the same order the info was registered.
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    for dup_name, t in duplicates:
        writer.write_tensor_data(t.data, tensor_endianess=src_reader.endianess)

    writer.close()


def quantize(
    input_gguf: Path,
    imatrix: Path,
    output_path: Path,
    lm_head_tensor_type: str,
    token_embedding_type: str,
    body_tensor_type: str,
    tied_embeddings: bool,
    pure: bool,
    tensor_type_overrides: list[str],
) -> None:
    cmd = [
        str(LLAMA_QUANTIZE_BIN),
        "--imatrix", str(imatrix),
    ]
    if tied_embeddings:
        # Only token_embd.weight exists (shared with lm_head). llama-quantize applies
        # --token-embedding-type to token_embd before its tied logic, so this deterministically
        # sets the single shared tensor to the lower-precision of the two configured types.
        shared_type = lower_precision_type(lm_head_tensor_type, token_embedding_type)
        print(f"  Tied embeddings: shared lm_head/token_embd tensor -> {shared_type}")
        cmd += ["--token-embedding-type", shared_type]
    else:
        cmd += [
            "--output-tensor-type", lm_head_tensor_type,
            "--token-embedding-type", token_embedding_type,
        ]
    # Per-tensor overrides: each "REGEX=TYPE" is matched (regex_search) against the
    # tensor name; matching tensors in the body use TYPE instead of body_tensor_type.
    # First match wins, and these apply even under --pure.
    for tt in tensor_type_overrides:
        cmd += ["--tensor-type", tt]
    if pure:
        # --pure disables llama.cpp's per-tensor quant mixtures so the body is quantized
        # exactly to body_tensor_type (e.g. no q4_0 -> q4_1 bump on early ffn_down layers).
        cmd += ["--pure"]
    cmd += [str(input_gguf), str(output_path), body_tensor_type.upper()]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a GGUF model: configurable body, lm_head and embedding types."
    )
    parser.add_argument("input_gguf", type=Path, help="Path to the input GGUF file (BF16/F16/F32)")
    parser.add_argument("imatrix", type=Path, help="Path to the importance matrix file")
    parser.add_argument("output", type=Path, help="Path for the output quantized GGUF file")
    parser.add_argument(
        "--body-tensor-type", default="q4_1", metavar="TYPE",
        help="Quant type for the model body (default: q4_1; e.g. q4_0)",
    )
    # When lm_head and token_embd are separate tensors (untied), each uses its own type below.
    # When they are shared (tied, detected via has_unique_lm_head), the single shared tensor
    # is quantized to the lower-precision of the two.
    parser.add_argument(
        "--lm-head-tensor-type", default="q8_0", metavar="TYPE",
        help="Quant type for lm_head / output.weight (default: q8_0)",
    )
    parser.add_argument(
        "--token-embedding-type", default="bf16", metavar="TYPE",
        help="Quant type for token embeddings (default: bf16)",
    )
    parser.add_argument(
        "--pure", action="store_true",
        help="Pass --pure to llama-quantize: disable per-tensor quant mixtures so the body "
             "is quantized exactly to --body-tensor-type (avoids e.g. q4_0 -> q4_1 on early "
             "ffn_down layers). lm_head/embedding overrides are still applied.",
    )
    parser.add_argument(
        "--tensor-type", action="append", default=[], dest="tensor_type", metavar="REGEX=TYPE",
        help="Override the quant type for body tensors whose name matches REGEX (regex_search) "
             "to TYPE, leaving the rest at --body-tensor-type. Repeatable; first match wins; "
             "applies even with --pure. e.g. --tensor-type attn_v=q8_0 "
             "--tensor-type 'blk\\.[0-3]\\.ffn_down=q6_k'",
    )
    parser.add_argument(
        "--duplicate-tensor", action="append", default=[], dest="duplicate_tensor",
        metavar="REGEX[=REPLACEMENT]",
        help="Keep an extra copy of every source tensor whose name matches REGEX (regex_search), "
             "in its ORIGINAL dtype, alongside the quantized version, for later debug/comparison. "
             "Without =REPLACEMENT the copy is named '<name>_duplicate'. With =REPLACEMENT the copy "
             "name is re.sub(REGEX, REPLACEMENT, name), so backreferences build unique per-tensor "
             "names. Repeatable; first matching spec wins. e.g. --duplicate-tensor attn_v "
             "--duplicate-tensor 'blk\\.(\\d+)\\.ssm_alpha\\.weight=blk.\\1.ssm_alpha.weight_dup'",
    )


    args = parser.parse_args()

    if not args.input_gguf.exists():
        raise FileNotFoundError(f"Input GGUF not found: {args.input_gguf}")
    if not args.imatrix.exists():
        raise FileNotFoundError(f"imatrix not found: {args.imatrix}")
    if not LLAMA_QUANTIZE_BIN.exists():
        raise FileNotFoundError(f"llama-quantize binary not found: {LLAMA_QUANTIZE_BIN}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".gguf", dir=args.output.parent)
    tmp_path = Path(tmp_name)
    try:
        os.close(tmp_fd)
        tied_embeddings = not has_unique_lm_head(args.input_gguf)
        print(f"Pass 1: quantize original bf16 → {args.body_tensor_type.upper()} (norm/bias remain F32)")
        quantize(
            args.input_gguf,
            args.imatrix,
            tmp_path,
            lm_head_tensor_type=args.lm_head_tensor_type,
            token_embedding_type=args.token_embedding_type,
            body_tensor_type=args.body_tensor_type,
            tied_embeddings=tied_embeddings,
            pure=args.pure,
            tensor_type_overrides=args.tensor_type,
        )
        # print("Pass 2: copy quantized file, converting F32 norm/bias → bf16")

        if args.duplicate_tensor:
            print("Pass 2: copy quantized file, appending original-dtype duplicate tensors")
            copy_with_duplicates(tmp_path, args.input_gguf, args.output, args.duplicate_tensor)
        else:
            import shutil
            shutil.copy2(tmp_path, args.output)
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
