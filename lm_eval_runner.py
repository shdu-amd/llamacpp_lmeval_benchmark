"""Live-scoring wrapper around lm-evaluation-harness.

lm-eval is the source of truth: prompts, few-shot assembly, answer extraction,
and scoring all come from the vendored harness (``lm-evaluation-harness/``), not
from us. This module only orchestrates it against an already-running
OpenAI-compatible server (llama.cpp) and reproduces the "running score every N
samples" UX the project wants.

How live scoring works without owning lm-eval's loop
----------------------------------------------------
lm-eval scores in one batch at the end of a run, so it has no notion of a running
score. We get one via **chunked re-runs backed by lm-eval's request cache**: we
call ``simple_evaluate`` repeatedly with a growing per-subtask ``limit``
(X, 2X, 3X, ...), all sharing one sqlite cache. Greedy generations are cached by
request content (``CachingLM``), so each chunk only *generates* the new X
docs/subtask; everything prior is served from cache. After each chunk we persist
lm-eval's full results dict as the current checkpoint. Scoring (cheap regex) is
redone each chunk; generation (expensive) is not.

Persistence philosophy
----------------------
We do **not** reshape lm-eval's output. Each checkpoint stores the complete
``simple_evaluate`` return verbatim under an ``lm_eval`` key, plus a thin
envelope of our own launch/progress metadata. Different tasks report different
metrics (exact_match vs acc vs acc_norm, with different meanings); by saving the
ground truth as-is we let separate, per-benchmark visualization code decide how
to interpret it. This runner stays benchmark-agnostic.

Only generative tasks (``output_type: generate_until``, e.g. mmlu_pro, gsm8k)
work over the chat endpoint. Loglikelihood/multiple-choice tasks need logprobs
and are rejected up front rather than scored meaninglessly.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Small helpers (formerly in bench_common)
# --------------------------------------------------------------------------- #
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def append_json_line(path: str, payload: Dict[str, Any]) -> None:
    """Append one checkpoint to a pretty-printed JSON array.

    Kept as a single valid, indented JSON document (not JSONL) so the file stays
    human-readable, while preserving the full checkpoint-to-checkpoint history
    for plotting score continuation across a run.
    """
    history: List[Any] = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = [history]
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(payload)
    atomic_write_json(path, history)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class LMEvalLiveRunner:
    """Drive one lm-eval task against a running server, checkpointing a running
    score every ``checkpoint_every`` docs-per-subtask."""

    # Output types that produce a text generation we can get over chat.
    _GENERATIVE = {"generate_until"}

    def __init__(
        self,
        task: str,
        base_url: str,
        model_name: str,
        run_dir: str,
        run_meta: Dict[str, Any],
        target_limit: Optional[int] = None,
        checkpoint_every: int = 25,
        num_concurrent: int = 1,
        gen_kwargs: str = "",
        request_timeout: float = 900.0,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        random_seed: int = 0,
        fewshot_seed: int = 1234,
    ):
        self.task = task
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.run_dir = run_dir
        self.run_meta = run_meta
        self.target_limit = target_limit
        self.checkpoint_every = max(1, checkpoint_every)
        self.num_concurrent = max(1, num_concurrent)
        self.gen_kwargs = gen_kwargs
        self.request_timeout = request_timeout
        self.max_retries = max(1, max_retries)
        self.api_key = api_key
        self.random_seed = random_seed
        self.fewshot_seed = fewshot_seed

        self.progress_path = os.path.join(run_dir, "progress.json")
        self.final_path = os.path.join(run_dir, "results.json")
        self.samples_dir = os.path.join(run_dir, "samples")
        self.cache_path = os.path.join(run_dir, "cache")
        self._start_ts = time.time()

        # Populated by _prepare().
        self._tm = None
        self._leaf_tasks: List[str] = []
        # Per-leaf shuffled doc-id order (seeded), built in _prepare(); a growing
        # prefix is fed to each chunk. None only before _prepare() runs.
        self._perms: Optional[Dict[str, List[int]]] = None
        # True when --limit was omitted and the target was auto-set to full.
        self._auto_full = False

    # ---- setup + guard ---------------------------------------------------- #
    def _prepare(self) -> None:
        import random

        from lm_eval.tasks import TaskManager

        self._tm = TaskManager(verbosity="WARNING")
        if self.task not in self._tm.task_index:
            raise SystemExit(
                f"error: unknown lm-eval task/group '{self.task}'. "
                f"List tasks with: lm_eval --tasks list"
            )
        # TaskManager.load() returns a flat {"tasks": {name: Task}, "groups":..},
        # even for a group like mmlu_pro; "tasks" is already the leaf mapping.
        leaves = self._tm.load(self.task)["tasks"]

        bad = {
            n: t.OUTPUT_TYPE for n, t in leaves.items()
            if t.OUTPUT_TYPE not in self._GENERATIVE
        }
        if bad:
            details = ", ".join(f"{n} ({t})" for n, t in sorted(bad.items()))
            raise SystemExit(
                f"error: task '{self.task}' has non-generative subtask(s) that need "
                f"loglikelihood/logprobs and cannot be scored over the chat "
                f"endpoint (local-chat-completions): {details}. "
                f"Use a generate_until task (e.g. mmlu_pro, gsm8k)."
            )

        self._leaf_tasks = list(leaves)

        # When no --limit is given, run the *full* dataset but still checkpoint:
        # the per-subtask target becomes the largest leaf's doc count, so the
        # ladder climbs high enough to cover every doc. Smaller leaves saturate
        # early (their shuffled prefix is capped at their own size below).
        leaf_sizes = {name: len(leaves[name].eval_docs) for name in leaves}
        if self.target_limit is None:
            self._auto_full = True
            self.target_limit = max(leaf_sizes.values(), default=0)

        # Each leaf gets its own seeded permutation of doc ids; a growing prefix
        # is fed to each chunk via `samples=`, so the running score is over a
        # random (not head-of-file) subset. One RNG seeded by random_seed, drawn
        # in sorted leaf order, keeps permutations reproducible across resume.
        rng = random.Random(self.random_seed)
        perms: Dict[str, List[int]] = {}
        for name in sorted(leaves):
            k = leaf_sizes[name]
            order = list(range(k))
            rng.shuffle(order)
            perms[name] = order[: min(self.target_limit, k)]
        self._perms = perms

    def _model_args(self) -> str:
        parts = [
            f"base_url={self.base_url}/v1/chat/completions",
            f"model={self.model_name}",
            f"num_concurrent={self.num_concurrent}",
            f"max_retries={self.max_retries}",
            "tokenized_requests=False",
        ]
        if self.request_timeout:
            parts.append(f"timeout={int(self.request_timeout)}")
        return ",".join(parts)

    # ---- one chunk -------------------------------------------------------- #
    def _run_chunk(self, limit: Optional[int]) -> Dict[str, Any]:
        import lm_eval

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if self.api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)
        else:
            # local-chat-completions still expects the var to exist.
            os.environ.setdefault("OPENAI_API_KEY", "sk-no-key-required")

        # When subsampling, feed a growing prefix of each leaf's shuffled doc
            # ids via `samples` (mutually exclusive with `limit`). Full runs also
            # use this path: "full" means the shuffled prefix reaches each leaf's
            # entire doc set, not native-order traversal via `limit=`.
        samples = None
        if self._perms is not None:
            samples = {name: ids[:limit] for name, ids in self._perms.items()}
            limit = None

        results = lm_eval.simple_evaluate(
            model="local-chat-completions",
            model_args=self._model_args(),
            tasks=[self.task],
            limit=limit,
            samples=samples,
            use_cache=self.cache_path,
            # The request-building cache keys on task/fewshot/template but NOT on
            # the sample set, and only rebuilds the full dataset on the `limit`
            # path. On the `samples` path (chunked live scoring) it caches just the
            # current prefix, so a later, longer chunk loads too few instances and
            # postprocessing IndexErrors on the missing docs. Skip it when
            # subsampling; generation is still cached content-wise via use_cache.
            cache_requests=False,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            gen_kwargs=self.gen_kwargs if self.gen_kwargs else None,
            log_samples=True,
            random_seed=self.random_seed,
            fewshot_random_seed=self.fewshot_seed,
            task_manager=self._tm,
            verbosity="WARNING",
        )
        return results

    # ---- persistence ------------------------------------------------------ #
    @staticmethod
    def _split_samples(results: Dict[str, Any]) -> Dict[str, Any]:
        """Pop the large per-doc ``samples`` out of the results dict.

        Returns the popped samples mapping; ``results`` is mutated in place so the
        checkpoint envelope stays small. Samples are written separately.
        """
        return results.pop("samples", {}) if isinstance(results, dict) else {}

    def _headline(self, results: Dict[str, Any]) -> Optional[str]:
        """Best-effort single-line metric for stdout, read generically."""
        res = (results or {}).get("results", {})
        node = res.get(self.task)
        if not isinstance(node, dict):
            # Single-task run: results is keyed by the leaf task name.
            if len(res) == 1:
                node = next(iter(res.values()))
        if not isinstance(node, dict):
            return None
        for key, val in node.items():
            if "," in key and isinstance(val, (int, float)) and not key.endswith(
                "_stderr"
            ):
                return f"{key}={val:.4f}"
        return None

    def _envelope(
        self, results: Dict[str, Any], status: str, current_limit: int
    ) -> Dict[str, Any]:
        n_subtasks = max(1, len(self._leaf_tasks))
        # Count real docs, not current_limit * n_subtasks: once a small leaf runs
        # out its prefix stops growing, so a flat multiply would overcount.
        if self._perms:
            total = sum(len(ids) for ids in self._perms.values())
            completed = sum(
                min(current_limit, len(ids)) for ids in self._perms.values()
            )
        else:
            total = (self.target_limit or 0) * n_subtasks if self.target_limit else None
            completed = current_limit * n_subtasks
        elapsed = time.time() - self._start_ts
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = ((total - completed) / rate) if (rate > 0 and total) else None
        return {
            "task": self.task,
            "run": self.run_meta,
            "status": status,
            "progress": {
                "completed": completed,
                "total": total,
                "current_limit_per_subtask": current_limit,
                "n_subtasks": n_subtasks,
                "elapsed_seconds": round(elapsed, 1),
                "samples_per_second": round(rate, 3),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "updated_at": utc_now_iso(),
            },
            "lm_eval": results,  # full harness output, verbatim (samples removed)
        }

    def _write_samples(self, samples: Dict[str, Any]) -> None:
        if not samples:
            return
        os.makedirs(self.samples_dir, exist_ok=True)
        for task_name, docs in samples.items():
            atomic_write_json(
                os.path.join(self.samples_dir, f"{task_name}.json"),
                {"task": task_name, "samples": docs},
            )

    # ---- main loop -------------------------------------------------------- #
    def run(self) -> Dict[str, Any]:
        self._prepare()
        os.makedirs(self.run_dir, exist_ok=True)
        atomic_write_json(
            os.path.join(self.run_dir, "run_meta.json"),
            {
                "task": self.task,
                "leaf_tasks": self._leaf_tasks,
                # Shuffled doc-id order per leaf. lm-eval re-enumerates sampled
                # docs from 0, so this is the only record mapping an output
                # sample back to its true dataset index.
                "sample_doc_ids": self._perms,
                **self.run_meta,
            },
        )

        n_subtasks = len(self._leaf_tasks)
        limit_desc = (
            f"full (auto, max {self.target_limit}/subtask)"
            if self._auto_full
            else f"{self.target_limit}/subtask"
        )
        print(
            f"[lm-eval] task={self.task} ({n_subtasks} subtask(s)), "
            f"limit/subtask={limit_desc}, "
            f"checkpoint every {self.checkpoint_every}/subtask, "
            f"num_concurrent={self.num_concurrent}",
            flush=True,
        )

        # Ladder of per-subtask limits: checkpoint_every, 2x, 3x, ..., target.
        # Each rung re-runs the growing prefix; generations are served from the
        # sqlite cache so only the newly added docs actually hit the server.
        ladder = list(
            range(self.checkpoint_every, self.target_limit + 1,
                  self.checkpoint_every)
        )
        if not ladder or ladder[-1] != self.target_limit:
            ladder.append(self.target_limit)

        last_results: Dict[str, Any] = {}
        current_limit = 0
        current_chunk = 0
        try:
            for i, limit in enumerate(ladder):
                is_last = i == len(ladder) - 1
                current_chunk = i + 1
                current_limit = limit or 0
                lbl = "full" if limit is None else f"{limit}/subtask"
                print(f"[lm-eval] chunk {i + 1}/{len(ladder)} -> limit={lbl}",
                      flush=True)
                results = self._run_chunk(limit) or {}
                samples = self._split_samples(results)
                last_results = results

                status = "done" if is_last else "running"
                env = self._envelope(results, status, current_limit)
                append_json_line(self.progress_path, env)
                if is_last:
                    self._write_samples(samples)
                    atomic_write_json(self.final_path, env)

                head = self._headline(results)
                head_str = f"  {head}" if head else ""
                print(f"[lm-eval] chunk {i + 1} done{head_str}", flush=True)
        except BaseException as e:
            env = self._envelope(last_results, "failed", current_limit)
            env["error"] = f"{type(e).__name__}: {e!r}"
            env["failed_chunk"] = {
                "index": current_chunk,
                "limit_per_subtask": current_limit,
                "total_chunks": len(ladder),
            }
            append_json_line(self.progress_path, env)
            raise

        return self._envelope(last_results, "done", ladder[-1] or 0)
