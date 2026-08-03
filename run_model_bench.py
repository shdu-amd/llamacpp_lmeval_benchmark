#!/usr/bin/env python3
"""Run an lm-evaluation-harness task against an already-running server.

This is the eval client. lm-eval is the source of truth for prompts, few-shot
assembly, answer extraction, and scoring (see ``lm_eval_runner.py``); this script
just parses CLI args and drives the runner. It is backend-agnostic:
point ``--base-url`` at any server that speaks ``/v1/chat/completions`` (llama.cpp
today) and it will:

  1. validate the task and reject non-generative ones (they need logprobs),
  2. run lm-eval in growing per-subtask chunks sharing one request cache, and
  3. checkpoint lm-eval's full results dict every N docs/subtask, so you can
     graph/visualize the running score while the run is still going.

Nothing here reshapes lm-eval's output: the checkpoint stores the harness result
verbatim under an ``lm_eval`` key. Interpretation (per-metric, per-benchmark) is
left to separate visualization code.

Launching the server is a separate step (see ``scripts/launch_llama_server.sh``),
so you can start a model once and run many sweeps without reloading weights.

Example:
    source scripts/active_environment.sh
    # shell 1:
    ./scripts/launch_llama_server.sh --model models/Qwen3.5-0.8B-Q4_1.gguf
    # shell 2:
    python run_model_bench.py \
        --base-url http://127.0.0.1:8080 \
        --model-name Qwen3.5-0.8B \
        --benchmark mmlu_pro \
        --limit 100          # optional: 100 docs *per subtask*
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from lm_eval_runner import LMEvalLiveRunner, utc_now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an lm-eval task on a running server, with a running score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", default=os.environ.get("LLAMACPP_BASE_URL"),
                   help="Server base URL, e.g. http://127.0.0.1:8080 "
                        "(default: $LLAMACPP_BASE_URL).")
    p.add_argument("--model-name", required=True,
                   help="Logical model name (sent as 'model' and recorded in results).")
    p.add_argument("--benchmark", required=True,
                   help="lm-eval task or group name (e.g. mmlu_pro, gsm8k). "
                        "Must be generative (output_type: generate_until).")
    p.add_argument("--output-dir", default="results",
                   help="Directory for checkpoint + final results JSON.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate N docs PER SUBTASK (lm-eval semantics). "
                        "Omit to cover the full task (auto target = largest "
                        "leaf's doc count), still checkpointed.")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Rewrite the rolling results JSON every N docs/subtask.")
    p.add_argument("--num-concurrent", type=int, default=1,
                   help="Parallel in-flight requests to the server. >1 needs the "
                        "server launched with matching --parallel slots.")

    # Generation params -> lm-eval gen_kwargs (greedy by default).
    g = p.add_argument_group("generation params")
    g.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy; >0 disables the cache).")
    g.add_argument("--max-tokens", type=int, default=2048,
                   help="Max tokens to generate (lm-eval max_gen_toks).")
    g.add_argument("--seed", type=int, default=1234,
                   help="Generation RNG seed.")
    g.add_argument("--top-k", type=int, default=None, help="Top-k cutoff (if supported).")
    g.add_argument("--top-p", type=float, default=None, help="Top-p cutoff (if supported).")

    p.add_argument("--random-seed", type=int, default=0,
                   help="lm-eval doc-shuffle seed.")
    p.add_argument("--fewshot-seed", type=int, default=1234,
                   help="lm-eval few-shot sampling seed.")
    p.add_argument("--request-timeout", type=float, default=600.0,
                   help="Per-request timeout (seconds).")
    p.add_argument("--max-retries", type=int, default=100,
                   help="Retries per request on transient failures (e.g. server "
                        "disconnects while the local server frees memory).")
    p.add_argument("--api-key", default=os.environ.get("LLAMACPP_API_KEY"),
                   help="Bearer token, if the server requires one.")
    return p.parse_args()


def _build_gen_kwargs(args: argparse.Namespace) -> str:
    parts = [
        f"temperature={args.temperature}",
        f"max_gen_toks={args.max_tokens}",
        f"seed={args.seed}",
    ]
    if args.top_k is not None:
        parts.append(f"top_k={args.top_k}")
    if args.top_p is not None:
        parts.append(f"top_p={args.top_p}")
    return ",".join(parts)


def main() -> int:
    args = parse_args()

    if not args.base_url:
        print("error: --base-url is required (or set $LLAMACPP_BASE_URL "
              "via scripts/active_environment.sh)", file=sys.stderr)
        return 2

    if args.temperature and args.temperature > 0:
        print("[run] WARNING: temperature > 0 disables lm-eval's request cache, "
              "so chunked live scoring will re-generate every chunk (slow).",
              file=sys.stderr, flush=True)
    if args.limit is None and args.checkpoint_every:
        print("[run] note: no --limit given -> full run over every doc, "
              "checkpointed every N docs/subtask up to the largest leaf.",
              flush=True)

    base_url = args.base_url.rstrip("/")
    model_name = args.model_name
    gen_kwargs = _build_gen_kwargs(args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.output_dir, args.benchmark, f"{model_name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    run_meta = {
        "model_name": model_name,
        "benchmark": args.benchmark,
        "base_url": base_url,
        "limit_per_subtask": args.limit,
        "checkpoint_every": args.checkpoint_every,
        "num_concurrent": args.num_concurrent,
        "gen_kwargs": gen_kwargs,
        "random_seed": args.random_seed,
        "fewshot_seed": args.fewshot_seed,
        "max_retries": args.max_retries,
        "started_at": utc_now_iso(),
    }

    print(f"[run] server:     {base_url}  (model={model_name})", flush=True)
    print(f"[run] task:       {args.benchmark}", flush=True)
    print(f"[run] output dir: {run_dir}", flush=True)
    print(f"[run] gen_kwargs: {gen_kwargs}", flush=True)

    runner = LMEvalLiveRunner(
        task=args.benchmark,
        base_url=base_url,
        model_name=model_name,
        run_dir=run_dir,
        run_meta=run_meta,
        target_limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        num_concurrent=args.num_concurrent,
        gen_kwargs=gen_kwargs,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        api_key=args.api_key,
        random_seed=args.random_seed,
        fewshot_seed=args.fewshot_seed,
    )
    try:
        final = runner.run()
    except SystemExit:
        raise
    except Exception as e:
        print(
            f"[run] FAILED: {type(e).__name__}: {e!r}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print("\n[run] done.", flush=True)
    head = runner._headline(final.get("lm_eval", {}))
    if head:
        print(f"[run] headline: {head}", flush=True)
    print(f"[run] full results: {os.path.join(run_dir, 'results.json')}", flush=True)
    print(f"[run] per-doc samples: {os.path.join(run_dir, 'samples')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
