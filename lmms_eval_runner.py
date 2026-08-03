"""Live-scoring driver for lmms-eval against a running OpenAI-compatible server.

lmms-eval owns everything that matters for correctness: prompts, few-shot
assembly, answer extraction and scoring. This module adds one thing on top —
a *running score* checkpointed every N docs/subtask, so a run can be graphed
while it is still in flight.

Mechanism
---------
lmms-eval scores once at the end, so we fake a running score by calling
``simple_evaluate`` on a growing per-subtask ``limit`` (N, 2N, 3N, ...). Every
call shares one on-disk response cache (``use_cache=<dir>``): a completed call
merges its greedy generations into ``<dir>/cache.db``, and the next call reads
them back, so each rung only *generates* the newly added N docs. Scoring (cheap
regex) is redone each rung; generation (expensive) is not. See
``lmms-eval/docs/advanced/caching.md``.

Two hard requirements make this reproducible and cheap:
  * greedy decoding (``temperature=0``, ``do_sample=false``) — the cache stores
    deterministic requests only, and greedy is what makes reruns identical;
  * fixed seeds — passed through to lmms-eval (python/numpy/torch + fewshot).

Only ``generate_until`` tasks work over a chat endpoint; loglikelihood tasks
need logprobs and are rejected up front.

Output layout (under ``run_dir``)::
    run_meta.json     launch config + leaf task sizes
    progress.json     JSON array, one checkpoint envelope per rung (live)
    results.json      final checkpoint envelope
    samples/<task>.json   per-doc inputs/outputs/scores (final rung only)
    cache/            lmms-eval response cache
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str, payload: Any) -> None:
    """Atomically write ``payload`` as indented JSON."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def _append_checkpoint(path: str, envelope: Dict[str, Any]) -> None:
    """Append one envelope to a JSON array file, kept valid and readable."""
    history: List[Any] = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
            history = loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(envelope)
    _write_json(path, history)


def _leaf_tasks(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten lmms-eval's (possibly nested / tuple-wrapped) task_dict to leaves.

    Groups come back as ``{group: {leaf: Task}}``; leaves may be wrapped as
    ``(group_name, Task)`` tuples. Returns a flat ``{leaf_name: Task}``.
    """
    leaves: Dict[str, Any] = {}
    for name, obj in task_dict.items():
        if isinstance(obj, dict):
            leaves.update(_leaf_tasks(obj))
        elif isinstance(obj, tuple):
            if obj[1] is not None:
                leaves[name] = obj[1]
        elif obj is not None:
            leaves[name] = obj
    return leaves


# Convenience aliases kept for external importers.
utc_now_iso = _utc_now


class LMMSEvalLiveRunner:
    """Run one lmms-eval task with a checkpointed running score."""

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
        self.gen_kwargs = gen_kwargs or ""
        self.request_timeout = request_timeout
        self.max_retries = max(1, max_retries)
        self.api_key = api_key
        self.random_seed = random_seed
        self.fewshot_seed = fewshot_seed

        self.progress_path = os.path.join(run_dir, "progress.json")
        self.results_path = os.path.join(run_dir, "results.json")
        self.samples_dir = os.path.join(run_dir, "samples")
        self.cache_dir = os.path.join(run_dir, "cache")

        self._task_manager = None
        self._leaf_sizes: Dict[str, int] = {}
        self._auto_full = False
        self._start_ts = time.time()

    # -- setup ------------------------------------------------------------- #
    def _prepare(self) -> None:
        """Load the task once to validate it and record per-leaf doc counts."""
        from lmms_eval.tasks import TaskManager, get_task_dict

        self._task_manager = TaskManager(verbosity="WARNING")
        if self.task not in self._task_manager.task_index:
            raise SystemExit(
                f"error: unknown lmms-eval task/group '{self.task}'. "
                f"List with: python -m lmms_eval --tasks list"
            )

        leaves = _leaf_tasks(get_task_dict([self.task], self._task_manager, "simple"))

        non_gen = {
            n: t.OUTPUT_TYPE for n, t in leaves.items()
            if t.OUTPUT_TYPE != "generate_until"
        }
        if non_gen:
            detail = ", ".join(f"{n} ({t})" for n, t in sorted(non_gen.items()))
            raise SystemExit(
                f"error: task '{self.task}' has non-generative subtask(s) needing "
                f"logprobs, which the chat endpoint cannot score: {detail}. "
                f"Use a generate_until task (e.g. mmlu_pro, gsm8k)."
            )

        self._leaf_sizes = {n: len(t.eval_docs) for n, t in leaves.items()}
        if self.target_limit is None:
            self._auto_full = True
            self.target_limit = max(self._leaf_sizes.values(), default=0)

    def _model_args(self) -> str:
        args = [
            f"model_version={self.model_name}",
            f"base_url={self.base_url}/v1",
            f"num_concurrent={self.num_concurrent}",
            f"max_retries={self.max_retries}",
        ]
        if self.request_timeout:
            args.append(f"timeout={int(self.request_timeout)}")
        return ",".join(args)

    def _ladder(self) -> List[int]:
        """Per-subtask limits: N, 2N, 3N, ..., target (last rung == target)."""
        step, target = self.checkpoint_every, self.target_limit
        rungs = list(range(step, target + 1, step))
        if not rungs or rungs[-1] != target:
            rungs.append(target)
        return rungs

    # -- one rung ---------------------------------------------------------- #
    def _evaluate(self, limit: int) -> Dict[str, Any]:
        from lmms_eval.evaluator import simple_evaluate

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OPENAI_API_KEY", self.api_key or "sk-no-key-required")

        return simple_evaluate(
            model="openai",
            model_args=self._model_args(),
            # Text tasks build their own prompt via doc_to_text + few-shot; the
            # simple OpenAI backend skips chat-message/multimodal plumbing.
            force_simple=True,
            tasks=[self.task],
            limit=limit,
            use_cache=self.cache_dir,
            cache_requests=False,
            apply_chat_template=False,
            gen_kwargs=self.gen_kwargs or None,
            log_samples=True,
            random_seed=self.random_seed,
            numpy_random_seed=self.random_seed,
            torch_random_seed=self.random_seed,
            fewshot_random_seed=self.fewshot_seed,
            task_manager=self._task_manager,
            verbosity="WARNING",
        ) or {}

    # -- checkpoint -------------------------------------------------------- #
    def _envelope(
        self, results: Dict[str, Any], status: str, limit: int
    ) -> Dict[str, Any]:
        total = sum(self._leaf_sizes.values()) or None
        done = sum(min(limit, size) for size in self._leaf_sizes.values())
        elapsed = time.time() - self._start_ts
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if (rate > 0 and total) else None
        return {
            "task": self.task,
            "run": self.run_meta,
            "status": status,
            "progress": {
                "completed": done,
                "total": total,
                "current_limit_per_subtask": limit,
                "n_subtasks": len(self._leaf_sizes),
                "elapsed_seconds": round(elapsed, 1),
                "samples_per_second": round(rate, 3),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "updated_at": _utc_now(),
            },
            "lmms_eval": results,  # full harness output, verbatim (samples split out)
        }

    def _write_samples(self, samples: Dict[str, Any]) -> None:
        if not samples:
            return
        os.makedirs(self.samples_dir, exist_ok=True)
        for task_name, docs in samples.items():
            _write_json(
                os.path.join(self.samples_dir, f"{task_name}.json"),
                {"task": task_name, "samples": docs},
            )

    def headline(self, results: Dict[str, Any]) -> Optional[str]:
        """A single ``metric,filter=value`` line for stdout, read generically."""
        node = (results or {}).get("results", {}).get(self.task)
        if not isinstance(node, dict):
            leaves = (results or {}).get("results", {})
            node = next(iter(leaves.values())) if len(leaves) == 1 else None
        if not isinstance(node, dict):
            return None
        for key, val in node.items():
            if "," in key and isinstance(val, (int, float)) and not key.endswith("_stderr"):
                return f"{key}={val:.4f}"
        return None

    # -- main -------------------------------------------------------------- #
    def run(self) -> Dict[str, Any]:
        self._prepare()
        os.makedirs(self.run_dir, exist_ok=True)
        _write_json(
            os.path.join(self.run_dir, "run_meta.json"),
            {"task": self.task, "leaf_sizes": self._leaf_sizes, **self.run_meta},
        )

        ladder = self._ladder()
        limit_desc = (
            f"full (auto, max {self.target_limit}/subtask)"
            if self._auto_full else f"{self.target_limit}/subtask"
        )
        print(
            f"[lmms-eval] task={self.task} ({len(self._leaf_sizes)} subtask(s)), "
            f"limit/subtask={limit_desc}, checkpoint every {self.checkpoint_every}, "
            f"num_concurrent={self.num_concurrent}",
            flush=True,
        )

        last: Dict[str, Any] = {}
        limit = 0
        try:
            for i, limit in enumerate(ladder):
                is_last = i == len(ladder) - 1
                print(f"[lmms-eval] chunk {i + 1}/{len(ladder)} -> limit={limit}/subtask",
                      flush=True)

                results = self._evaluate(limit)
                samples = results.pop("samples", {}) if isinstance(results, dict) else {}
                last = results

                envelope = self._envelope(results, "done" if is_last else "running", limit)
                _append_checkpoint(self.progress_path, envelope)
                if is_last:
                    self._write_samples(samples)
                    _write_json(self.results_path, envelope)

                head = self.headline(results)
                print(f"[lmms-eval] chunk {i + 1} done" + (f"  {head}" if head else ""),
                      flush=True)
        except BaseException as e:
            envelope = self._envelope(last, "failed", limit)
            envelope["error"] = f"{type(e).__name__}: {e!r}"
            _append_checkpoint(self.progress_path, envelope)
            raise

        return self._envelope(last, "done", ladder[-1])
