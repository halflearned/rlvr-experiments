#!/usr/bin/env python3
"""
Analyze GRPO stalls on the Hendrycks MATH dataset.

Typical usage:
  .venv/bin/python scripts/adhoc/analyze_math_grpo_stall.py \
    --config configs/qwen3-1.7B-math-lr1e6.yaml \
    --trace traces/trace_20260113_134506.jsonl \
    --rollouts traces/rollouts_20260113_134506.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.ipc as ipc
import yaml
from transformers import AutoTokenizer


SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@dataclass(frozen=True)
class PromptInfo:
    prompt_id: str
    level: str
    subject: str
    template_tokens: int


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _iter_hf_cached_math_rows(
    *,
    hf_datasets_cache: Path,
    subject: str,
    split: str,
) -> Iterable[dict]:
    """
    Stream rows from a locally cached HuggingFace datasets Arrow file.

    Cache layout (HF datasets):
      ~/.cache/huggingface/datasets/EleutherAI___hendrycks_math/<subject>/0.0.0/<hash>/hendrycks_math-<split>.arrow

    The `.arrow` files are Arrow IPC *streams* (use `ipc.open_stream`).
    """
    subject_root = hf_datasets_cache / "EleutherAI___hendrycks_math" / subject / "0.0.0"
    if not subject_root.exists():
        raise FileNotFoundError(f"Missing HF cache for subject={subject}: {subject_root}")

    arrow_path = None
    for hash_dir in sorted([p for p in subject_root.iterdir() if p.is_dir()]):
        candidate = hash_dir / f"hendrycks_math-{split}.arrow"
        if candidate.exists():
            arrow_path = candidate
            break
    if arrow_path is None:
        raise FileNotFoundError(f"Missing cached arrow for subject={subject} split={split} under {subject_root}")

    with pa.memory_map(str(arrow_path), "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            schema = batch.schema
            problems = batch.column(schema.get_field_index("problem"))
            levels = batch.column(schema.get_field_index("level"))
            types = batch.column(schema.get_field_index("type"))
            solutions = batch.column(schema.get_field_index("solution"))
            for i in range(batch.num_rows):
                yield {
                    "problem": problems[i].as_py(),
                    "level": levels[i].as_py(),
                    "type": types[i].as_py(),
                    "solution": solutions[i].as_py(),
                }


def _make_template(
    *,
    tokenizer,
    raw_problem: str,
    system_prompt: str,
    assistant_prefix: str,
) -> str:
    prompt = f"\n\nProblem:{raw_problem.strip()}"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    content = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return content + (assistant_prefix or "")


def find_too_long_math_prompts(
    *,
    config: dict,
    hf_datasets_cache: Path,
) -> tuple[list[PromptInfo], dict]:
    tokenizer_cfg = config.get("tokenizer", {})
    tokenizer_path = tokenizer_cfg.get("pretrained_model_name_or_path") or config.get("model", {}).get("path")
    if not tokenizer_path:
        raise ValueError("Could not determine tokenizer path from config.")

    data_cfg = config.get("data", {})
    levels = data_cfg.get("level") or []
    level_strs = {f"Level {l}" for l in levels} if levels else None

    data_iter_cfg = config.get("data_iter", {})
    system_prompt = data_iter_cfg.get("system_prompt", "")
    assistant_prefix = data_iter_cfg.get("assistant_prefix", "")

    rollout_role = next((r for r in config.get("roles", []) if r.get("name") == "rollout"), None)
    if not rollout_role:
        raise ValueError("Could not find a 'rollout' role in config.roles.")
    max_model_len = rollout_role.get("config", {}).get("max_model_len")
    if not isinstance(max_model_len, int):
        raise ValueError("Could not determine rollout.config.max_model_len from config.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=bool(tokenizer_cfg.get("use_fast", False)))

    too_long: list[PromptInfo] = []
    max_item: PromptInfo | None = None
    stats = Counter()

    split = data_cfg.get("split", "train")
    for subject in SUBJECTS:
        for row in _iter_hf_cached_math_rows(hf_datasets_cache=hf_datasets_cache, subject=subject, split=split):
            if level_strs is not None and row["level"] not in level_strs:
                continue

            raw_problem = row["problem"]
            prompt = f"\n\nProblem:{raw_problem.strip()}"
            prompt_id = f"math_{row['type']}_{_hash_prompt(prompt)}"

            template = _make_template(
                tokenizer=tokenizer,
                raw_problem=raw_problem,
                system_prompt=system_prompt,
                assistant_prefix=assistant_prefix,
            )
            template_tokens = len(tokenizer(template, add_special_tokens=False).input_ids)

            stats["rows_considered"] += 1
            if max_item is None or template_tokens > max_item.template_tokens:
                max_item = PromptInfo(
                    prompt_id=prompt_id,
                    level=row["level"],
                    subject=row["type"],
                    template_tokens=template_tokens,
                )

            if template_tokens >= max_model_len:
                too_long.append(
                    PromptInfo(
                        prompt_id=prompt_id,
                        level=row["level"],
                        subject=row["type"],
                        template_tokens=template_tokens,
                    )
                )

    meta = {
        "tokenizer_path": tokenizer_path,
        "split": split,
        "level_filter": sorted(level_strs) if level_strs is not None else None,
        "rollout_max_model_len": max_model_len,
        "rows_considered": stats["rows_considered"],
        "max_item": max_item.__dict__ if max_item is not None else None,
    }
    return too_long, meta


def _analyze_trace(trace_path: Path) -> dict:
    last_metrics_ts = -1.0
    last_metrics = None
    last_buf = None

    span_counts = Counter()
    counter_counts = Counter()

    for line in trace_path.open("r"):
        obj = json.loads(line)
        et = obj.get("type")
        if et == "counter":
            counter_counts[obj.get("name")] += 1
            if obj.get("name") == "metrics":
                last_metrics_ts = obj.get("ts", -1.0)
                last_metrics = obj
        elif et == "span":
            span_counts[obj.get("name")] += 1
        elif et == "buffer":
            last_buf = obj

    spans_after = Counter()
    counters_after = Counter()
    for line in trace_path.open("r"):
        obj = json.loads(line)
        if obj.get("ts", -1.0) <= last_metrics_ts:
            continue
        if obj.get("type") == "span":
            spans_after[obj.get("name")] += 1
        if obj.get("type") == "counter":
            counters_after[obj.get("name")] += 1

    return {
        "last_metrics_ts": last_metrics_ts,
        "last_metrics": last_metrics,
        "last_buffer": last_buf,
        "span_counts": dict(span_counts),
        "counter_counts": dict(counter_counts),
        "spans_after_last_metrics": dict(spans_after),
        "counters_after_last_metrics": dict(counters_after),
    }


def _analyze_rollouts(rollouts_path: Path) -> dict:
    unique = set()
    zero_var = 0
    total = 0
    for line in rollouts_path.open("r"):
        obj = json.loads(line)
        total += 1
        pid = obj.get("prompt_id")
        if pid:
            unique.add(pid)
        rewards = obj.get("rewards") or []
        if rewards and (max(rewards) - min(rewards) == 0):
            zero_var += 1
    return {"lines": total, "unique_prompt_ids": len(unique), "zero_variance_reward_rollouts": zero_var}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Training config YAML (e.g. configs/qwen3-1.7B-math-lr1e6.yaml)")
    ap.add_argument("--trace", required=True, help="Trace JSONL (e.g. traces/trace_*.jsonl)")
    ap.add_argument("--rollouts", required=True, help="Rollouts JSONL (e.g. traces/rollouts_*.jsonl)")
    ap.add_argument(
        "--hf-datasets-cache",
        default=str(Path.home() / ".cache" / "huggingface" / "datasets"),
        help="HF datasets cache root (read-only OK).",
    )
    args = ap.parse_args()

    config = _load_yaml(args.config)
    trace_path = Path(args.trace)
    rollouts_path = Path(args.rollouts)
    hf_cache = Path(args.hf_datasets_cache)

    trace_info = _analyze_trace(trace_path)
    rollouts_info = _analyze_rollouts(rollouts_path)
    too_long, too_long_meta = find_too_long_math_prompts(config=config, hf_datasets_cache=hf_cache)

    rollouts_prompt_ids = set()
    with rollouts_path.open("r") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj.get("prompt_id")
            if pid:
                rollouts_prompt_ids.add(pid)

    print("\n=== Trace Summary ===")
    print(f"trace: {trace_path}")
    print(f"last metrics ts: {trace_info['last_metrics_ts']}")
    print(f"last metrics: {trace_info['last_metrics']}")
    if trace_info["last_buffer"] is not None:
        last_buf = trace_info["last_buffer"]
        fates = last_buf.get("fates") or {}
        used = sum((fates.get("used") or {}).values())
        wasted = sum((fates.get("wasted") or {}).values())
        filtered = sum((fates.get("filtered") or {}).values())
        failed = sum((fates.get("failed") or {}).values())
        print(
            f"last buffer: ts={last_buf.get('ts')} size={last_buf.get('size')} "
            f"used={used} wasted={wasted} filtered={filtered} failed={failed}"
        )
    else:
        print("last buffer: (none)")

    print("\nspans after last metrics:")
    for name, c in sorted(trace_info["spans_after_last_metrics"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  {name}: {c}")

    print("\ncounters after last metrics:")
    for name, c in sorted(trace_info["counters_after_last_metrics"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  {name}: {c}")

    print("\n=== Rollouts Summary ===")
    print(f"rollouts: {rollouts_path}")
    print(json.dumps(rollouts_info, indent=2))

    print("\n=== Cached MATH Prompt Lengths (Template Tokens) ===")
    print(json.dumps(too_long_meta, indent=2))
    if not too_long:
        print("No prompts exceed rollout max_model_len.")
        return 0

    too_long_sorted = sorted(too_long, key=lambda x: x.template_tokens, reverse=True)
    print(f"\nPrompts with template_tokens >= rollout_max_model_len ({too_long_meta['rollout_max_model_len']}):")
    for pi in too_long_sorted:
        in_rollouts = pi.prompt_id in rollouts_prompt_ids
        print(
            f"- {pi.prompt_id} level={pi.level} subject={pi.subject} template_tokens={pi.template_tokens} "
            f"in_rollouts={in_rollouts}"
        )

    missing = [pi for pi in too_long_sorted if pi.prompt_id not in rollouts_prompt_ids]
    if missing:
        print("\nMissing from rollouts (likely stuck before log_rollout):")
        for pi in missing:
            print(f"- {pi.prompt_id} template_tokens={pi.template_tokens}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
