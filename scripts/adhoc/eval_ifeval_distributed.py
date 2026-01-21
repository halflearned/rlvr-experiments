#!/usr/bin/env python3
"""Distributed full-dataset IFEval evaluation (streaming, multi-GPU).

This script shards RLVR-IFeval across GPUs, runs vLLM generation in each worker,
verifies instruction-following constraints, and writes per-prompt counts to JSONL.

Unlike `scripts/eval_pass_rate.py`, this avoids materializing all completions in RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

from rlvr_experiments.verifiers.ifeval import VALIDATION_FUNCTIONS


@dataclass(frozen=True)
class PromptRecord:
    idx: int
    prompt_id: str
    prompt: str
    ground_truth: str
    constraint_type: str


def _parse_ground_truth(ground_truth: str) -> tuple[str | None, dict]:
    try:
        gt = json.loads(ground_truth)
    except Exception:
        return None, {}
    func_name = gt.get("func_name")
    kwargs = {k: v for k, v in gt.items() if k != "func_name" and v is not None}
    return func_name, kwargs


def pass_at_k_prob(n: int, c: int, k: int) -> float:
    """P(at least 1 correct in k draws without replacement from n, with c correct)."""
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0
    # P(all wrong) = C(n-c, k) / C(n, k)
    # Use multiplicative form for numerical stability / speed.
    p_all_wrong = 1.0
    for i in range(k):
        p_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - p_all_wrong


def _load_records(split: str, num_prompts: int, seed: int) -> list[PromptRecord]:
    ds = load_dataset("allenai/RLVR-IFeval", split=split)
    records: list[PromptRecord] = []
    for i, row in enumerate(ds):
        user_content = row["messages"][0]["content"] if row.get("messages") else ""
        records.append(
            PromptRecord(
                idx=i,
                prompt_id=f"ifeval_{i}",
                prompt=user_content,
                ground_truth=row["ground_truth"],
                constraint_type=row["constraint_type"],
            )
        )

    if num_prompts <= 0:
        return records

    import random

    random.seed(seed)
    return random.sample(records, min(num_prompts, len(records)))


def _shard_contiguous(records: list[PromptRecord], shards: int) -> list[list[PromptRecord]]:
    if shards <= 0:
        raise ValueError("shards must be > 0")
    n = len(records)
    base = n // shards
    rem = n % shards
    out: list[list[PromptRecord]] = []
    start = 0
    for s in range(shards):
        size = base + (1 if s < rem else 0)
        out.append(records[start : start + size])
        start += size
    return out


def _read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Ignore a trailing partial line from an interrupted write.
                continue
            pid = obj.get("prompt_id")
            if pid:
                done.add(pid)
    return done


def _worker_main(
    *,
    worker_id: int,
    gpu_id: int,
    records: list[PromptRecord],
    model_path: str,
    sampling_params_dict: dict,
    max_model_len: int,
    enforce_eager: bool,
    batch_size: int,
    output_jsonl: str,
    progress_queue,
    resume: bool,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if resume:
        done_ids = _read_done_ids(out_path)

    sampling_params = SamplingParams(**sampling_params_dict)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        enable_prefix_caching=True,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
    )

    total = len(records)
    processed = 0
    t0 = time.time()

    progress_queue.put((worker_id, processed, total, time.time() - t0))
    # Line-buffered writes so progress is durable.
    with out_path.open("a", buffering=1) as f:
        batch: list[PromptRecord] = []
        for rec in records:
            if rec.prompt_id in done_ids:
                processed += 1
                continue
            batch.append(rec)
            if len(batch) < batch_size:
                continue

            _process_batch(batch, llm, sampling_params, f)
            processed += len(batch)
            batch.clear()
            if processed % 25 == 0 or processed == total:
                progress_queue.put((worker_id, processed, total, time.time() - t0))

        if batch:
            _process_batch(batch, llm, sampling_params, f)
            processed += len(batch)
            batch.clear()
            progress_queue.put((worker_id, processed, total, time.time() - t0))


def _process_batch(
    batch: list[PromptRecord],
    llm: LLM,
    sampling_params: SamplingParams,
    file_handle,
) -> None:
    prompts = [r.prompt for r in batch]
    outputs = llm.generate(prompts, sampling_params)

    for rec, out in zip(batch, outputs):
        func_name, kwargs = _parse_ground_truth(rec.ground_truth)
        func = VALIDATION_FUNCTIONS.get(func_name) if func_name else None

        num_completions = len(out.outputs)
        num_correct = 0
        if func is not None:
            for o in out.outputs:
                try:
                    ok = func(o.text, **kwargs)
                except Exception:
                    ok = False
                if ok:
                    num_correct += 1

        file_handle.write(
            json.dumps(
                {
                    "idx": rec.idx,
                    "prompt_id": rec.prompt_id,
                    "constraint_type": rec.constraint_type,
                    "num_completions": num_completions,
                    "num_correct": num_correct,
                }
            )
            + "\n"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed RLVR-IFeval pass@k evaluation")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num-prompts", type=int, default=0, help="0 or -1 means all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=512, help="Completions per prompt (max k)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile/CUDA graph paths in vLLM (slower but often more stable).",
    )
    p.add_argument("--batch-size", type=int, default=2, help="Prompts per vLLM generate call, per worker")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--replicate",
        action="store_true",
        help="Have every GPU process the full dataset (useful to build n_total=GPUs×n and merge).",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Stable identifier for output filenames (useful for restarts). "
             "If omitted, defaults to a timestamp.",
    )
    p.add_argument("--resume", action="store_true", help="Resume from existing worker JSONL files")
    p.add_argument(
        "--watchdog-seconds",
        type=float,
        default=1800.0,
        help="Restart a worker if no progress update is received for this long (0 disables).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IFEval Distributed Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Prompts: {args.num_prompts if args.num_prompts > 0 else 'all'}")
    print(f"n (max k): {args.n}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"GPUs: {gpu_ids}")
    print(f"Enforce eager: {args.enforce_eager}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output dir: {output_dir}")
    print(f"Run id: {args.run_id or '<timestamp>'}")
    print(f"Replicate: {args.replicate}")
    print(f"Resume: {args.resume}")
    print(f"Watchdog seconds: {args.watchdog_seconds}")
    print("=" * 60)

    records = _load_records(args.split, args.num_prompts, args.seed)
    print(f"Loaded {len(records)} prompts")

    if args.replicate:
        shards = [records for _ in gpu_ids]
    else:
        shards = _shard_contiguous(records, shards=len(gpu_ids))

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    progress_q = ctx.Queue()
    procs = []
    worker_files = []

    sampling_params_dict = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "n": args.n,
        # Ensure different runs can generate different samples deterministically.
        # Individual workers can override this by passing a different seed.
        "seed": args.seed,
    }

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    for worker_id, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
        out_jsonl = output_dir / f"ifeval_worker{worker_id:02d}_gpu{gpu_id}_n{args.n}_{run_id}.jsonl"
        worker_files.append(out_jsonl)
        # Derive per-worker seed so replicate mode yields distinct samples.
        worker_sampling_params = dict(sampling_params_dict)
        worker_sampling_params["seed"] = int(args.seed) + int(worker_id)
        p = ctx.Process(
            target=_worker_main,
            kwargs={
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "records": shard,
                "model_path": args.model_path,
                "sampling_params_dict": worker_sampling_params,
                "max_model_len": args.max_model_len,
                "enforce_eager": args.enforce_eager,
                "batch_size": args.batch_size,
                "output_jsonl": str(out_jsonl),
                "progress_queue": progress_q,
                "resume": args.resume,
            },
        )
        procs.append(p)
        p.start()

    totals = {wid: len(shard) for wid, shard in enumerate(shards)}
    done = {wid: 0 for wid in range(len(shards))}
    last_update = {wid: time.monotonic() for wid in range(len(shards))}
    start = time.time()

    # Progress loop: wait for all workers to report reaching their totals.
    while True:
        all_finished = all(done[wid] >= totals[wid] for wid in done)
        if all_finished:
            break
        try:
            wid, processed, total, elapsed = progress_q.get(timeout=30.0)
        except queue.Empty:
            if args.watchdog_seconds and args.watchdog_seconds > 0:
                now = time.monotonic()
                stalled = [
                    wid
                    for wid in done
                    if done[wid] < totals[wid] and (now - last_update[wid]) > args.watchdog_seconds
                ]
                if stalled:
                    print(
                        f"[watchdog] no progress for {args.watchdog_seconds:.0f}s from workers: {stalled}",
                        flush=True,
                    )
                    for wid in stalled:
                        _restart_worker(
                            wid=wid,
                            ctx=ctx,
                            procs=procs,
                            gpu_ids=gpu_ids,
                            shards=shards,
                            model_path=args.model_path,
                            sampling_params_dict=sampling_params_dict,
                            max_model_len=args.max_model_len,
                            batch_size=args.batch_size,
                            worker_files=worker_files,
                            progress_q=progress_q,
                            resume=True,
                        )
                        last_update[wid] = now
            # Periodic liveness check.
            for i, p in enumerate(procs):
                if p.exitcode is not None and p.exitcode != 0:
                    raise RuntimeError(f"Worker {i} exited with code {p.exitcode}")
            continue
        done[wid] = max(done.get(wid, 0), processed)
        last_update[wid] = time.monotonic()
        overall_done = sum(done.values())
        overall_total = sum(totals.values())
        wall = time.time() - start
        rate = overall_done / wall if wall > 0 else 0.0
        remaining = (overall_total - overall_done) / rate if rate > 0 else float("inf")
        print(
            f"[progress] {overall_done}/{overall_total} prompts "
            f"({overall_done/overall_total*100:.1f}%) "
            f"rate={rate:.2f} prompts/s ETA={remaining/60:.0f}m"
            ,
            flush=True,
        )

    for p in procs:
        p.join()
    for i, p in enumerate(procs):
        if p.exitcode != 0:
            raise RuntimeError(f"Worker {i} exited with code {p.exitcode}")

    wall_time = time.time() - start
    print(f"Workers finished in {wall_time/60:.1f} minutes, aggregating...", flush=True)

    merged_by_prompt_id: dict[str, dict] = {}
    for path in worker_files:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = r.get("prompt_id")
                if not pid:
                    continue
                if pid not in merged_by_prompt_id:
                    merged_by_prompt_id[pid] = {
                        "idx": r.get("idx"),
                        "prompt_id": pid,
                        "constraint_type": r.get("constraint_type", "unknown"),
                        "num_completions": 0,
                        "num_correct": 0,
                    }
                merged_by_prompt_id[pid]["num_completions"] += int(r.get("num_completions", 0))
                merged_by_prompt_id[pid]["num_correct"] += int(r.get("num_correct", 0))

    results = list(merged_by_prompt_id.values())
    if not results:
        raise RuntimeError("No results found; worker output files are empty/missing.")

    # Aggregate metrics
    total_prompts = len(results)
    total_completions = sum(r.get("num_completions", 0) for r in results)
    total_correct = sum(r.get("num_correct", 0) for r in results)

    overall_pass_rate = total_correct / total_completions if total_completions else 0.0

    # In replicate mode, effective n is GPUs×n; otherwise it's just n.
    max_n = args.n * len(gpu_ids) if args.replicate else args.n
    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, max_n]
    pass_at_k = {}
    for k in k_values:
        probs = [pass_at_k_prob(r["num_completions"], r["num_correct"], k) for r in results]
        pass_at_k[k] = sum(probs) / len(probs)

    # Breakdown by constraint type
    by_constraint = defaultdict(lambda: {"prompts": 0, "completions": 0, "correct": 0})
    any_correct_by_constraint = Counter()
    for r in results:
        ct = r.get("constraint_type", "unknown")
        by_constraint[ct]["prompts"] += 1
        by_constraint[ct]["completions"] += r["num_completions"]
        by_constraint[ct]["correct"] += r["num_correct"]
        if r["num_correct"] > 0:
            any_correct_by_constraint[ct] += 1

    by_constraint_out = {}
    for ct, agg in by_constraint.items():
        by_constraint_out[ct] = {
            "num_prompts": agg["prompts"],
            "pass_rate": agg["correct"] / agg["completions"] if agg["completions"] else 0.0,
            "pass_at_n": any_correct_by_constraint[ct] / agg["prompts"] if agg["prompts"] else 0.0,
        }

    # Write summary
    summary_path = output_dir / f"ifeval_summary_{Path(args.model_path).name}_dp{len(gpu_ids)}_n{args.n}_{run_id}.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "metadata": {
                    "dataset": "ifeval",
                    "split": args.split,
                    "model_path": args.model_path,
                    "num_prompts": total_prompts,
                    "n_per_worker": args.n,
                    "n_total": max_n,
                    "max_tokens": args.max_tokens,
                    "max_model_len": args.max_model_len,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "gpus": gpu_ids,
                    "batch_size": args.batch_size,
                    "wall_time_seconds": wall_time,
                    "run_id": run_id,
                    "replicate": args.replicate,
                },
                "summary": {
                    "total_completions": total_completions,
                    "total_correct": total_correct,
                    "overall_pass_rate": overall_pass_rate,
                    "pass_at_k": pass_at_k,
                    "by_constraint_type": by_constraint_out,
                },
            },
            f,
            indent=2,
        )

    print(f"Summary written to: {summary_path}", flush=True)
    print("Pass@k:")
    for k in sorted(pass_at_k):
        print(f"  pass@{k}: {pass_at_k[k]*100:.2f}%")
    print(f"Overall pass rate (per completion): {overall_pass_rate*100:.2f}%")
    return 0


def _kill_process_tree(pid: int) -> None:
    try:
        from vllm.utils.system_utils import kill_process_tree
        kill_process_tree(pid)
    except Exception:
        try:
            os.kill(pid, 9)
        except Exception:
            pass


def _restart_worker(
    *,
    wid: int,
    ctx,
    procs: list,
    gpu_ids: list[int],
    shards: list[list[PromptRecord]],
    model_path: str,
    sampling_params_dict: dict,
    max_model_len: int,
    enforce_eager: bool,
    batch_size: int,
    worker_files: list[Path],
    progress_q,
    resume: bool,
) -> None:
    old = procs[wid]
    if old.is_alive():
        try:
            if old.pid is not None:
                _kill_process_tree(old.pid)
        finally:
            old.join(timeout=10)

    gpu_id = gpu_ids[wid]
    out_jsonl = worker_files[wid]
    shard = shards[wid]

    p = ctx.Process(
        target=_worker_main,
        kwargs={
            "worker_id": wid,
            "gpu_id": gpu_id,
            "records": shard,
            "model_path": model_path,
            "sampling_params_dict": sampling_params_dict,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            "batch_size": batch_size,
            "output_jsonl": str(out_jsonl),
            "progress_queue": progress_q,
            "resume": resume,
        },
    )
    procs[wid] = p
    p.start()


if __name__ == "__main__":
    raise SystemExit(main())
