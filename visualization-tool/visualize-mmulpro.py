#!/usr/bin/env python3
"""Plot per-category MMLU-Pro accuracy across live-scoring checkpoints.

Point this at a run directory (or any parent of one) containing a ``progress.json``
written by ``lmms_eval_runner`` -- a JSON array of checkpoint envelopes. For each
category it draws two views over the checkpoint ladder (x-axis = samples scored
per category so far):

  * Cumulative accuracy: the score lmms-eval reports at each checkpoint, taken
    over *all* samples seen up to that point. This is the raw per-category number.
  * Incremental accuracy: the score contributed by *only* the new samples added
    between consecutive checkpoints. lmms-eval never reports this; we derive it by
    differencing cumulative correct-counts.

So yes -- the per-category score in progress.json is cumulative, not per-chunk.
The incremental panel recovers the per-checkpoint signal.

Usage:
    python visualize-mmulpro.py [PATH] [-o OUT.png] [--show]

PATH defaults to the newest run under ./results and may be either the run dir
itself or a parent to search.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def find_progress(path: str) -> str:
    """Resolve a user-supplied path to a concrete progress.json file."""
    if os.path.isfile(path):
        return path
    direct = os.path.join(path, "progress.json")
    if os.path.isfile(direct):
        return direct
    matches = glob.glob(os.path.join(path, "**", "progress.json"), recursive=True)
    if not matches:
        raise SystemExit(f"error: no progress.json found under '{path}'")
    # Newest by mtime so a bare parent dir resolves to the latest run.
    return max(matches, key=os.path.getmtime)


def load_checkpoints(progress_path: str) -> List[Dict[str, Any]]:
    with open(progress_path) as f:
        data = json.load(f)
    if isinstance(data, dict):  # tolerate a single-object (pre-array) file
        data = [data]
    if not data:
        raise SystemExit(f"error: '{progress_path}' has no checkpoints")
    return data


def _metric_key(node: Dict[str, Any]) -> Optional[str]:
    """Pick the primary accuracy metric key generically (e.g. exact_match,...)."""
    for key, val in node.items():
        if "," in key and isinstance(val, (int, float)) and not key.endswith(
            "_stderr"
        ):
            return key
    return None


def build_series(
    checkpoints: List[Dict[str, Any]],
) -> Tuple[List[int], Dict[str, List[Optional[float]]], Dict[str, List[Optional[float]]]]:
    """Return (x_samples, cumulative, incremental) keyed by category alias.

    x_samples is the per-category sample count at each checkpoint. Missing values
    (a category absent from a checkpoint) are None so lines break cleanly.
    """
    x_samples: List[int] = []
    cumulative: Dict[str, List[Optional[float]]] = {}
    # Cumulative correct-count per category, used to difference into incremental.
    cum_correct: Dict[str, List[Optional[float]]] = {}
    cum_n: Dict[str, List[Optional[int]]] = {}

    # Drop group/aggregate nodes (e.g. the overall "mmlu_pro" roll-up); keep only
    # the leaf subject categories.
    group_names = set(checkpoints[0].get("lmms_eval", {}).get("groups", {}))
    categories: List[str] = []
    # alias -> leaf task name, so we can look up per-task counts in n-samples.
    alias_to_task: Dict[str, str] = {}
    for name, node in checkpoints[0].get("lmms_eval", {}).get("results", {}).items():
        if name in group_names:
            continue
        alias = node.get("alias", node.get("name", "?"))
        categories.append(alias)
        alias_to_task[alias] = name

    for ckpt in checkpoints:
        prog = ckpt.get("progress", {})
        cur_limit = int(prog.get("current_limit_per_subtask", 0))
        x_samples.append(cur_limit)
        lmms = ckpt.get("lmms_eval", {})
        results = lmms.get("results", {})
        # lmms-eval reports per-task doc counts here (per-node sample_len is gone).
        n_samples = lmms.get("n-samples", {})
        by_alias = {
            n.get("alias", n.get("name", "?")): (nm, n)
            for nm, n in results.items()
            if nm not in group_names
        }
        for cat in categories:
            entry = by_alias.get(cat)
            if entry is None:
                cumulative.setdefault(cat, []).append(None)
                cum_correct.setdefault(cat, []).append(None)
                cum_n.setdefault(cat, []).append(None)
                continue
            task_name, node = entry
            key = _metric_key(node)
            score = float(node[key]) if key else None
            eff = n_samples.get(task_name, {}).get("effective")
            n = int(eff) if eff is not None else cur_limit
            cumulative.setdefault(cat, []).append(score)
            cum_correct.setdefault(cat, []).append(
                score * n if score is not None else None
            )
            cum_n.setdefault(cat, []).append(n)

    incremental: Dict[str, List[Optional[float]]] = {}
    for cat in categories:
        correct = cum_correct[cat]
        ns = cum_n[cat]
        inc: List[Optional[float]] = []
        for i in range(len(correct)):
            if correct[i] is None or ns[i] is None:
                inc.append(None)
                continue
            if i == 0:
                dn, dc = ns[i], correct[i]
            else:
                prev_c = correct[i - 1] if correct[i - 1] is not None else 0.0
                prev_n = ns[i - 1] if ns[i - 1] is not None else 0
                dn, dc = ns[i] - prev_n, correct[i] - prev_c
            inc.append((dc / dn) if dn > 0 else None)
        incremental[cat] = inc

    # Once a category exhausts its docs, its sample count plateaus (the sampling
    # prefix caps at the category's size) while others keep growing. Mask those
    # no-growth checkpoints so the line stops where the category ran out instead
    # of drawing a misleading flat segment.
    for cat in categories:
        ns = cum_n[cat]
        for i in range(1, len(ns)):
            if ns[i] is not None and ns[i - 1] is not None and ns[i] <= ns[i - 1]:
                cumulative[cat][i] = None

    return x_samples, cumulative, incremental


def _macro_avg(
    series: Dict[str, List[Optional[float]]], n: int, carry_forward: bool = False
) -> List[Optional[float]]:
    """Mean across categories at each checkpoint.

    With ``carry_forward`` (cumulative view), a category that has run out of new
    samples keeps contributing its last known score, so the overall line reflects
    every category. Without it (incremental view), gaps are simply skipped.
    """
    last: Dict[str, float] = {}
    out: List[Optional[float]] = []
    for i in range(n):
        vals: List[float] = []
        for cat, v in series.items():
            if v[i] is not None:
                if carry_forward:
                    last[cat] = v[i]
                vals.append(v[i])
            elif carry_forward and cat in last:
                vals.append(last[cat])
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def _label_line(ax, x, ys, text, color):
    """Annotate a line at its last non-empty point so each is self-identifying."""
    for i in range(len(ys) - 1, -1, -1):
        if ys[i] is not None:
            ax.annotate(text, xy=(x[i], ys[i]), xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=7, color=color, clip_on=False)
            return


def _plot_panel(ax, x, series, title, colors, macro_carry=False):
    for (cat, ys), color in zip(sorted(series.items()), colors):
        ax.plot(x, ys, marker="o", markersize=4, linewidth=1.4,
                color=color, label=cat)
    macro = _macro_avg(series, len(x), carry_forward=macro_carry)
    ax.plot(x, macro, marker="s", markersize=5, linewidth=2.6, color="black",
            linestyle="--", label="macro avg")
    _label_line(ax, x, macro, "macro avg", "black")
    ax.set_title(title)
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default="results",
                        help="run dir, a progress.json, or a parent to search "
                             "(default: newest run under ./results)")
    parser.add_argument("-o", "--out", default=None,
                        help="output image path (default: <run>/mmlu_pro_progress.png)")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args(argv)

    progress_path = find_progress(args.path)
    checkpoints = load_checkpoints(progress_path)
    x, cumulative, incremental = build_series(checkpoints)

    task = checkpoints[0].get("task", "mmlu_pro")
    run = checkpoints[0].get("run", {})
    model = run.get("model_name", "?")
    status = checkpoints[-1].get("status", "?")

    n_cat = len(cumulative)
    colors = plt.cm.tab20(range(n_cat)) if n_cat <= 20 else plt.cm.gist_ncar(
        [i / max(1, n_cat - 1) for i in range(n_cat)]
    )

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    _plot_panel(ax_top, x, cumulative, "Cumulative accuracy (over all samples so far)", colors, macro_carry=True)
    _plot_panel(ax_bot, x, incremental, "Incremental accuracy (new samples per checkpoint)", colors)
    ax_bot.set_xlabel("number of samples per category (checkpoint x samples/checkpoint)")
    # Room on the right for the end-of-line category labels.
    span = (x[-1] - x[0]) or 1
    ax_bot.set_xlim(x[0] - 0.02 * span, x[-1] + 0.18 * span)

    fig.suptitle(f"{task} - {model}  [{status}]  ({n_cat} categories)", fontsize=13)
    # Prominent shared legend to the right so every colored line is identifiable.
    handles, labels = ax_top.get_legend_handles_labels()
    fig.subplots_adjust(right=0.8)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.81, 0.5),
               fontsize=9, ncol=1, title="category", title_fontsize=11,
               frameon=True, handlelength=2.4, markerscale=1.4,
               labelspacing=0.7, borderpad=0.8)
    fig.tight_layout(rect=(0, 0, 0.8, 0.97))

    out = args.out or os.path.join(os.path.dirname(progress_path),
                                   "mmlu_pro_progress.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"read   {progress_path} ({len(checkpoints)} checkpoints, {n_cat} categories)")
    print(f"wrote  {out}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
