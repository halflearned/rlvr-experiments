#!/usr/bin/env python3
"""Generate D3.js staleness comparison plots for the writeup.

Usage:
    python generate_staleness_plots.py config.yaml

Config format (YAML):
    output: staleness_plots.js
    runs:
      - path: /path/to/trace1.jsonl
        label: "max_staleness=0"
        color: "#3b82f6"
      - path: /path/to/trace2.jsonl
        label: "max_staleness=1"
        color: "#22c55e"
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import yaml

CACHE_DIR = Path(__file__).parent / ".plot_cache"


def _cache_key(trace_path: str) -> str:
    """Cache key based on path, mtime, and size."""
    st = os.stat(trace_path)
    raw = f"{trace_path}:{st.st_mtime_ns}:{st.st_size}"
    return hashlib.md5(raw.encode()).hexdigest()


def extract_metrics_cached(trace_path: str) -> tuple[list[dict], dict]:
    """Extract metrics with file-based caching."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(trace_path)
    cache_file = CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        print(f"  (cached) {trace_path}", file=sys.stderr)
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    result = extract_metrics(trace_path)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


def extract_metrics(trace_path: str) -> tuple[list[dict], dict]:
    """Extract metrics from trace file, returning (per-step dicts, raw events for summaries)."""
    reward_stats_events = []
    grpo_debug_events = []
    metrics_events = []
    batch_padding_events = []
    titan_metrics_events = []
    batch_lag_events = []
    skipped_events = []
    retry_events = []
    epoch_events = []

    with open(trace_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") != "counter":
                continue
            name = obj.get("name")
            if name == "reward_stats":
                reward_stats_events.append(obj)
            elif name == "grpo.debug":
                grpo_debug_events.append(obj)
            elif name == "metrics":
                metrics_events.append(obj)
            elif name == "batch.padding":
                batch_padding_events.append(obj)
            elif name == "titan.metrics":
                titan_metrics_events.append(obj)
            elif name == "batch.lag":
                batch_lag_events.append(obj)
            elif name == "skipped":
                skipped_events.append(obj)
            elif name == "retry":
                retry_events.append(obj)
            elif name in ("epoch", "epoch_complete"):
                epoch_events.append(obj)

    # Use batch.padding as the anchor since it has step numbers
    step_to_padding = {e["step"]: e for e in batch_padding_events if "step" in e}

    def find_nearest(events: list[dict], target_ts: float, tolerance: float = 5.0):
        """Find event with timestamp closest to target, within tolerance."""
        best = None
        best_dist = float("inf")
        for e in events:
            dist = abs(e["ts"] - target_ts)
            if dist < best_dist and dist < tolerance:
                best = e
                best_dist = dist
        return best

    results = []

    if step_to_padding:
        # Standard RL path: use batch.padding as step anchor
        steps = sorted(step_to_padding.keys())

        for step in steps:
            padding = step_to_padding[step]
            ts = padding["ts"]

            record = {"step": step}

            # Find matching events
            reward_stat = find_nearest(reward_stats_events, ts)
            if reward_stat:
                record["reward_overall"] = reward_stat.get("reward_overall")
                record["reward_used"] = reward_stat.get("reward_used")
                record["frac_all_correct"] = reward_stat.get("frac_all_correct")
                record["frac_all_wrong"] = reward_stat.get("frac_all_wrong")

            grpo_debug = find_nearest(grpo_debug_events, ts)
            if grpo_debug:
                record["kl_mean"] = grpo_debug.get("kl_mean")
                record["entropy_mean"] = grpo_debug.get("entropy_mean")

            metrics = find_nearest(metrics_events, ts)
            if metrics:
                record["loss"] = metrics.get("loss")
                record["loss_grpo"] = metrics.get("loss_grpo")
                record["loss_sft"] = metrics.get("loss_sft")

            # Compute mean completion length from batch.padding
            comp_lens = padding.get("completion_lens", [])
            if comp_lens:
                record["completion_len"] = sum(comp_lens) / len(comp_lens)

            results.append(record)
    elif metrics_events:
        # SFT-only path: no batch.padding, use metrics events sequentially
        for step_idx, m in enumerate(metrics_events, 1):
            record = {"step": step_idx}
            record["loss"] = m.get("loss")
            record["loss_sft"] = m.get("loss")  # surface as SFT loss too
            results.append(record)

    # Return raw events for summary computation
    raw_events = {
        "titan_metrics": titan_metrics_events,
        "batch_lag": batch_lag_events,
        "skipped": skipped_events,
        "retry": retry_events,
        "epoch": epoch_events,
    }

    return results, raw_events


def compute_summary(metrics: list[dict], raw_events: dict, run_id: str) -> dict:
    """Compute summary statistics for a run."""
    summary = {"id": run_id}

    # Helper to get last N valid values and compute mean
    def last_n_mean(key: str, n: int = 10) -> float | None:
        vals = [m.get(key) for m in metrics if m.get(key) is not None]
        if not vals:
            return None
        return sum(vals[-n:]) / min(n, len(vals))

    # Helper to get final value
    def final_val(key: str) -> float | None:
        for m in reversed(metrics):
            if m.get(key) is not None:
                return m[key]
        return None

    # Final values
    summary["final_reward"] = final_val("reward_overall")
    summary["final_loss"] = final_val("loss")
    summary["final_kl"] = final_val("kl_mean")
    summary["final_entropy"] = final_val("entropy_mean")
    summary["final_completion_len"] = final_val("completion_len")
    summary["final_frac_all_correct"] = final_val("frac_all_correct")
    summary["final_frac_all_wrong"] = final_val("frac_all_wrong")

    # Last 10-step averages (smoother)
    summary["avg_reward"] = last_n_mean("reward_overall", 10)
    summary["avg_loss"] = last_n_mean("loss", 10)
    summary["avg_completion_len"] = last_n_mean("completion_len", 10)

    # Total steps
    if metrics:
        summary["total_steps"] = metrics[-1].get("step", len(metrics))
    else:
        summary["total_steps"] = 0

    # === New metrics from raw events ===

    # Median MFU over steps 50-120 (stable range, excludes warmup and sync outliers)
    titan_events = raw_events.get("titan_metrics", [])
    if titan_events:
        # Use steps 50-120 if available, otherwise use what we have
        stable_range = titan_events[50:120] if len(titan_events) > 50 else titan_events
        mfu_vals = sorted([e.get("mfu") for e in stable_range if e.get("mfu") is not None])
        if mfu_vals:
            mid = len(mfu_vals) // 2
            summary["median_mfu"] = mfu_vals[mid] if len(mfu_vals) % 2 else (mfu_vals[mid-1] + mfu_vals[mid]) / 2
        else:
            summary["median_mfu"] = None
    else:
        summary["median_mfu"] = None

    # Total training time (from first epoch start to last epoch_complete)
    epoch_events = raw_events.get("epoch", [])
    start_ts = None
    end_ts = None
    for e in epoch_events:
        if e.get("name") == "epoch" and start_ts is None:
            start_ts = e.get("ts")
        if e.get("name") == "epoch_complete":
            end_ts = e.get("ts")
    if start_ts is not None and end_ts is not None:
        summary["total_training_time"] = end_ts - start_ts
    else:
        summary["total_training_time"] = None

    # Average lag breakdown from batch.lag events
    lag_events = raw_events.get("batch_lag", [])
    if lag_events:
        # Collect all lag_N keys and compute averages
        lag_counts = {}
        for e in lag_events:
            for k, v in e.items():
                if k.startswith("lag_") and isinstance(v, (int, float)):
                    lag_key = k  # e.g., "lag_0", "lag_1"
                    if lag_key not in lag_counts:
                        lag_counts[lag_key] = []
                    lag_counts[lag_key].append(v)
        # Store average for each lag bucket
        for lag_key, vals in lag_counts.items():
            summary[f"avg_{lag_key}"] = sum(vals) / len(vals) if vals else 0
        # Also store mean_lag average
        mean_lags = [e.get("mean_lag") for e in lag_events if e.get("mean_lag") is not None]
        summary["avg_mean_lag"] = sum(mean_lags) / len(mean_lags) if mean_lags else None
    else:
        summary["avg_mean_lag"] = None

    # Total filtered samples (skipped due to zero variance)
    skipped_events = raw_events.get("skipped", [])
    summary["total_filtered"] = sum(e.get("zero_variance", 0) for e in skipped_events)

    # Total wasted samples (stale evictions)
    retry_events = raw_events.get("retry", [])
    summary["total_wasted"] = sum(e.get("stale_evicted", 0) for e in retry_events)

    # Median time per step (from titan.metrics timestamp deltas, steps 50-120)
    if titan_events and len(titan_events) > 2:
        # Compute deltas between consecutive events
        timestamps = [e.get("ts") for e in titan_events if e.get("ts") is not None]
        deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        # Use steps 50-120 if available
        stable_deltas = sorted(deltas[50:120]) if len(deltas) > 50 else sorted(deltas)
        if stable_deltas:
            mid = len(stable_deltas) // 2
            summary["median_time_per_step"] = stable_deltas[mid] if len(stable_deltas) % 2 else (stable_deltas[mid-1] + stable_deltas[mid]) / 2
        else:
            summary["median_time_per_step"] = None
    else:
        summary["median_time_per_step"] = None

    return summary


def generate_plot_data(config: dict) -> dict:
    """Extract data from all runs and return structured plot data."""
    runs_data = []
    summaries = {}

    for run in config["runs"]:
        label = run["label"]
        color = run.get("color", "#666")
        run_id = run.get("id", label.replace("=", "_").replace(" ", "_"))

        # Support both run_dir (preferred) and explicit path/evals
        run_dir = run.get("run_dir")
        if run_dir:
            trace_path = Path(run_dir) / "traces" / "trace.jsonl"
            evals_path = Path(run_dir) / "evals" / "summary.json"
        else:
            trace_path = Path(run["path"])
            evals_path = Path(run["evals"]) if run.get("evals") else None

        print(f"Extracting from {trace_path}...", file=sys.stderr)
        metrics, raw_events = extract_metrics_cached(str(trace_path))

        # Compute summary for inline metrics
        summaries[run_id] = compute_summary(metrics, raw_events, run_id)
        summaries[run_id]["label"] = label
        summaries[run_id]["color"] = color

        # Load benchmark eval results
        summaries[run_id]["benchmarks"] = {}
        eval_curves = {}  # bench_name -> sorted list of (step, accuracy)
        if run_dir:
            evals_dir = Path(run_dir) / "evals"
            if evals_dir.exists():
                for bench_dir in sorted(evals_dir.iterdir()):
                    if not bench_dir.is_dir():
                        continue
                    bench_name = bench_dir.name
                    # Normalize ifbench -> ifbench_test (same benchmark, different dir names)
                    if bench_name == "ifbench":
                        bench_name = "ifbench_test"
                    summary_jsonl = bench_dir / "summary.jsonl"
                    summary_json = bench_dir / "summary.json"
                    # Helper: extract accuracy from eval entry
                    # Supports both "accuracy" (math/gsm8k) and
                    # "prompt_level_strict_acc" (ifeval/ifbench)
                    def _get_acc(entry):
                        return entry.get("accuracy") or entry.get("prompt_level_strict_acc")

                    def _store_benchmark(entry, step):
                        """Store benchmark data, splitting ifeval/ifbench into prompt/inst rows."""
                        acc = _get_acc(entry)
                        inst_acc = entry.get("inst_level_acc")
                        if acc is not None:
                            if inst_acc is not None:
                                # Split into prompt-level and inst-level rows
                                summaries[run_id]["benchmarks"][f"{bench_name} (prompt)"] = {
                                    "accuracy": acc, "step": step,
                                }
                                summaries[run_id]["benchmarks"][f"{bench_name} (inst)"] = {
                                    "accuracy": inst_acc, "step": step,
                                }
                            else:
                                summaries[run_id]["benchmarks"][bench_name] = {
                                    "accuracy": acc, "step": step,
                                }

                    if summary_jsonl.exists():
                        step_to_acc = {}
                        with open(summary_jsonl) as f:
                            for line in f:
                                entry = json.loads(line)
                                acc = _get_acc(entry)
                                step = entry.get("step")
                                if step is not None and acc is not None:
                                    step_to_acc[step] = acc
                                    if step == 200:
                                        _store_benchmark(entry, 200)
                        if step_to_acc:
                            entries = sorted(step_to_acc.items())
                            eval_curves[bench_name] = entries
                    elif summary_json.exists():
                        with open(summary_json) as f:
                            entry = json.load(f)
                        _store_benchmark(entry, entry.get("step"))
                    else:
                        # Fallback: check step*/summary.json subdirectories
                        step_to_acc = {}
                        for step_dir in sorted(bench_dir.glob("step*")):
                            sj = step_dir / "summary.json"
                            if sj.exists():
                                with open(sj) as f:
                                    entry = json.load(f)
                                acc = _get_acc(entry)
                                try:
                                    step = int(step_dir.name.replace("step", ""))
                                except ValueError:
                                    continue
                                if acc is not None:
                                    step_to_acc[step] = acc
                                    if step == 200:
                                        _store_benchmark(entry, 200)
                        if step_to_acc:
                            entries = sorted(step_to_acc.items())
                            eval_curves[bench_name] = entries

        run_entry = {
            "id": run_id,
            "label": label,
            "color": color,
            "steps": [m["step"] for m in metrics],
            "reward_overall": [m.get("reward_overall") for m in metrics],
            "frac_all_correct": [m.get("frac_all_correct") for m in metrics],
            "frac_all_wrong": [m.get("frac_all_wrong") for m in metrics],
            "completion_len": [m.get("completion_len") for m in metrics],
            "loss": [m.get("loss") for m in metrics],
            "loss_grpo": [m.get("loss_grpo") for m in metrics],
            "loss_sft": [m.get("loss_sft") for m in metrics],
            "kl_mean": [m.get("kl_mean") for m in metrics],
            "entropy_mean": [m.get("entropy_mean") for m in metrics],
        }

        # Add eval curves as eval_<bench>_steps / eval_<bench>_accuracy
        # (base model step-0 anchor is prepended in JS using base_model.benchmarks)
        for bench_name, entries in eval_curves.items():
            run_entry[f"eval_{bench_name}_steps"] = [s for s, _ in entries]
            run_entry[f"eval_{bench_name}_accuracy"] = [a for _, a in entries]

        # Load pass@k curves from merged_summary.json files
        if run_dir:
            evals_dir = Path(run_dir) / "evals"
            if evals_dir.exists():
                for bench_dir in sorted(evals_dir.iterdir()):
                    if not bench_dir.is_dir():
                        continue
                    bench_name = bench_dir.name

                    # Two possible layouts:
                    # 1. {bench}_pass{k}/step200/merged_summary.json (e.g. gsm8k_pass128)
                    # 2. {bench}/pass-at-k/merged_summary.json (e.g. ifeval/pass-at-k/)
                    merged = None
                    if "_pass" in bench_name:
                        merged = bench_dir / "step200" / "merged_summary.json"
                    else:
                        passk_dir = bench_dir / "pass-at-k"
                        if passk_dir.exists():
                            merged = passk_dir / "merged_summary.json"
                            if not merged.exists():
                                merged = passk_dir / "summary.json"
                            # Use mapped name to match base model convention
                            # Will be set after reading max_k below

                    if merged is None or not merged.exists():
                        continue
                    with open(merged) as f:
                        pdata = json.load(f)
                    pass_at_k = pdata.get("pass_at_k", {})
                    if pass_at_k:
                        # For pass-at-k/ subdirectory layout, map name to {bench}_pass{max_k}
                        if "_pass" not in bench_name:
                            max_k = max(int(key.split("@")[1]) for key in pass_at_k)
                            bench_name = f"{bench_name}_pass{max_k}"
                        # Store as passk_<bench_name>_k and passk_<bench_name>_rate
                        ks = []
                        rates = []
                        for key in sorted(pass_at_k.keys(), key=lambda x: int(x.split("@")[1])):
                            ks.append(int(key.split("@")[1]))
                            rates.append(pass_at_k[key])
                        run_entry[f"passk_{bench_name}_k"] = ks
                        run_entry[f"passk_{bench_name}_rate"] = rates

        runs_data.append(run_entry)

    # Load base model evals if configured
    base_benchmarks = {}
    base_config = config.get("base_model")
    if base_config:
        evals_dir = Path(base_config["evals_dir"])
        if evals_dir.exists():
            for bench_dir in sorted(evals_dir.iterdir()):
                if not bench_dir.is_dir():
                    continue
                bench_name = bench_dir.name
                summary_json = bench_dir / "summary.json"
                if summary_json.exists():
                    with open(summary_json) as f:
                        entry = json.load(f)
                    if "accuracy" in entry:
                        base_benchmarks[bench_name] = entry["accuracy"]
                    elif "prompt_level_strict_acc" in entry:
                        base_benchmarks[f"{bench_name} (prompt)"] = entry["prompt_level_strict_acc"]
                        if "inst_level_acc" in entry:
                            base_benchmarks[f"{bench_name} (inst)"] = entry["inst_level_acc"]

    # Load base model pass@k curves
    base_passk = {}
    if base_config:
        evals_dir = Path(base_config["evals_dir"])
        if evals_dir.exists():
            for bench_dir in sorted(evals_dir.iterdir()):
                if not bench_dir.is_dir():
                    continue
                passk_dir = bench_dir / "pass-at-k"
                merged = passk_dir / "merged_summary.json"
                if not merged.exists():
                    merged = passk_dir / "summary.json"
                if not merged.exists():
                    continue
                with open(merged) as f:
                    pdata = json.load(f)
                pass_at_k = pdata.get("pass_at_k", {})
                if pass_at_k:
                    bench_name = bench_dir.name
                    # Map base model bench names to the trained model pass@k key names
                    # e.g. "gsm8k" -> "gsm8k_pass128", "math" -> "math_pass128"
                    # Try to match by finding the max k
                    max_k = max(int(key.split("@")[1]) for key in pass_at_k)
                    mapped_name = f"{bench_name}_pass{max_k}"
                    ks = []
                    rates = []
                    for key in sorted(pass_at_k.keys(), key=lambda x: int(x.split("@")[1])):
                        ks.append(int(key.split("@")[1]))
                        rates.append(pass_at_k[key])
                    base_passk[mapped_name] = {"k": ks, "rate": rates}

    result = {"runs": runs_data, "summaries": summaries}
    if base_benchmarks:
        result["base_model"] = {
            "label": base_config.get("label", "Base"),
            "benchmarks": base_benchmarks,
        }
        if base_passk:
            result["base_model"]["passk"] = base_passk
    if config.get("exclude_benchmarks"):
        result["exclude_benchmarks"] = config["exclude_benchmarks"]
    return result


def generate_js(plot_data: dict, presets: list[dict] | None = None, html_prefix: str = "") -> str:
    """Generate D3.js code for the plots.

    If html_prefix is set, all DOM element IDs and global variable names are
    prefixed so multiple experiment panels can coexist on the same page.
    E.g., html_prefix="gsm8k" -> element IDs like "gsm8k-preset-buttons",
    JS globals like "gsm8kData", "gsm8kPresets".
    """
    data_json = json.dumps(plot_data, indent=2)
    presets_json = json.dumps(presets or [], indent=2)

    # Derive variable and ID prefixes
    p = html_prefix  # short alias for template use
    var_prefix = p.replace("-", "_") if p else "staleness"
    id_prefix = f"{p}-" if p else ""

    # Auto-discover pass@k benchmarks from the data
    passk_benchmarks = set()
    exclude_prefixes = plot_data.get("exclude_benchmarks", [])
    for run in plot_data.get("runs", []):
        for key in run:
            if key.startswith("passk_") and key.endswith("_k"):
                bench = key[len("passk_"):-len("_k")]  # e.g. "gsm8k_pass128"
                passk_benchmarks.add(bench)
    if plot_data.get("base_model", {}).get("passk"):
        for bench in plot_data["base_model"]["passk"]:
            passk_benchmarks.add(bench)

    # Build PassKConfigs entries, filtering by exclude_benchmarks
    # Dedup: if multiple pass@k variants exist for same base benchmark (e.g. aime_pass32
    # and aime_pass128), keep the one with highest max_k (most data points)
    BENCH_DISPLAY = {
        "gsm8k": "GSM8K", "math": "MATH", "aime": "AIME", "beyondaime": "BeyondAIME",
        "ifeval": "IFEval (prompt)", "ifbench": "IFBench (prompt)", "ifbench_test": "IFBench (prompt)",
        "humaneval": "HumanEval", "mbpp": "MBPP",
    }
    best_per_base = {}  # base_name -> (max_k, full_bench_name)
    for bench in passk_benchmarks:
        base_name = bench.rsplit("_pass", 1)[0]
        max_k = int(bench.rsplit("_pass", 1)[1])
        if base_name not in best_per_base or max_k > best_per_base[base_name][0]:
            best_per_base[base_name] = (max_k, bench)

    passk_configs = []
    for base_name in sorted(best_per_base):
        if any(base_name == ep or base_name.startswith(ep + " ") for ep in exclude_prefixes):
            continue
        _, bench = best_per_base[base_name]
        display = BENCH_DISPLAY.get(base_name, base_name.upper())
        passk_configs.append({
            "id": f"{id_prefix}plot-passk-{base_name}",
            "title": f"{display} Pass@k",
            "kKey": f"passk_{bench}_k",
            "rateKey": f"passk_{bench}_rate",
            "format": ".1%",
        })
    passk_configs_json = json.dumps(passk_configs, indent=2)

    js_code = f"""// Auto-generated experiment comparison plots
// Generated by generate_staleness_plots.py
// Prefix: {p or '(none)'}

const {var_prefix}Data = {data_json};
const {var_prefix}Presets = {presets_json};

// Format specifications for different metric types
const {var_prefix}Formats = {{
    'final_reward': {{ format: '.3f', suffix: '' }},
    'avg_reward': {{ format: '.3f', suffix: '' }},
    'final_loss': {{ format: '.4f', suffix: '' }},
    'avg_loss': {{ format: '.4f', suffix: '' }},
    'final_kl': {{ format: '.2e', suffix: '' }},
    'final_entropy': {{ format: '.3f', suffix: '' }},
    'final_completion_len': {{ format: '.0f', suffix: ' tokens' }},
    'avg_completion_len': {{ format: '.0f', suffix: ' tokens' }},
    'final_frac_all_correct': {{ format: '.1%', suffix: '' }},
    'final_frac_all_wrong': {{ format: '.1%', suffix: '' }},
    'total_steps': {{ format: 'd', suffix: '' }},
    'median_mfu': {{ format: '.1f', suffix: '%' }},
    'total_training_time': {{ format: '.0f', suffix: 's' }},
    'avg_mean_lag': {{ format: '.2f', suffix: '' }},
    'avg_lag_0': {{ format: '.1f', suffix: '' }},
    'avg_lag_1': {{ format: '.1f', suffix: '' }},
    'avg_lag_2': {{ format: '.1f', suffix: '' }},
    'total_filtered': {{ format: ',d', suffix: '' }},
    'total_wasted': {{ format: ',d', suffix: '' }},
    'median_time_per_step': {{ format: '.1f', suffix: 's' }},
    'eval_accuracy': {{ format: '.1%', suffix: '' }},
}};

// Populate inline metric spans for this section
function {var_prefix}PopulateMetrics() {{
    const spans = document.querySelectorAll('span.metric[data-run][data-key]');
    spans.forEach(span => {{
        const runId = span.dataset.run;
        const key = span.dataset.key;
        const summary = {var_prefix}Data.summaries[runId];
        if (!summary) return;  // not our section
        const value = summary[key];
        if (value === null || value === undefined) {{
            span.textContent = 'N/A';
            return;
        }}
        const fmt = {var_prefix}Formats[key] || {{ format: '.3f', suffix: '' }};
        span.textContent = d3.format(fmt.format)(value) + fmt.suffix;
        if (summary.color) {{
            span.style.color = summary.color;
            span.style.fontWeight = '600';
        }}
    }});
}}

// Generate comparison table
function {var_prefix}GenerateTable() {{
    const container = document.getElementById('{id_prefix}comparison-table');
    if (!container) return;

    const runs = {var_prefix}Data.runs;
    const summaries = {var_prefix}Data.summaries;
    const baseModel = {var_prefix}Data.base_model || null;

    const benchSet = new Set();
    const excludePrefixes = {var_prefix}Data.exclude_benchmarks || [];
    const shouldExclude = (b) => excludePrefixes.some(p => b === p || b.startsWith(p + ' '));
    runs.forEach(run => {{
        const s = summaries[run.id];
        if (s && s.benchmarks) Object.keys(s.benchmarks).forEach(b => {{ if (!shouldExclude(b)) benchSet.add(b); }});
    }});
    if (baseModel) Object.keys(baseModel.benchmarks).forEach(b => {{ if (!shouldExclude(b)) benchSet.add(b); }});
    const benchmarks = Array.from(benchSet).sort();
    if (benchmarks.length === 0) return;

    let html = '<table class="results-table comparison-table">';
    html += '<thead><tr><th>Benchmark (step 200)</th>';
    if (baseModel) html += `<th style="color:#999">${{baseModel.label}}</th>`;
    runs.forEach(run => {{
        html += `<th data-run-id="${{run.id}}" style="color: ${{run.color}}">${{run.label}}</th>`;
    }});
    html += '</tr></thead><tbody>';

    benchmarks.forEach(bench => {{
        html += `<tr><td>${{bench.toUpperCase()}}</td>`;
        if (baseModel) {{
            const bv = baseModel.benchmarks[bench];
            html += bv !== undefined && bv !== null
                ? `<td>${{d3.format('.1%')(bv)}}</td>`
                : '<td>&mdash;</td>';
        }}
        runs.forEach(run => {{
            const s = summaries[run.id];
            const bdata = s && s.benchmarks && s.benchmarks[bench];
            if (bdata && bdata.accuracy !== null && bdata.accuracy !== undefined) {{
                html += `<td data-run-id="${{run.id}}">${{d3.format('.1%')(bdata.accuracy)}}</td>`;
            }} else {{
                html += `<td data-run-id="${{run.id}}">&mdash;</td>`;
            }}
        }});
        html += '</tr>';
    }});

    html += '</tbody></table>';
    container.innerHTML = html;
}}

// Shared tooltip (one per page)
if (typeof window._plotTooltip === 'undefined') {{
    window._plotTooltip = null;
}}
function {var_prefix}GetTooltip() {{
    if (!window._plotTooltip) {{
        window._plotTooltip = d3.select('body').append('div')
            .attr('class', 'plot-tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.85)')
            .style('color', '#fff')
            .style('padding', '8px 12px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('font-family', 'Source Code Pro, monospace')
            .style('pointer-events', 'none')
            .style('opacity', 0)
            .style('z-index', 1000)
            .style('white-space', 'nowrap');
    }}
    return window._plotTooltip;
}}

// --- Preset & checkbox UI ---

const {var_prefix}Visible = new Set({var_prefix}Data.runs.map(r => r.id));

function {var_prefix}BuildPresets() {{
    const container = document.getElementById('{id_prefix}preset-buttons');
    if (!container || {var_prefix}Presets.length === 0) return;

    {var_prefix}Presets.forEach((preset, idx) => {{
        const btn = document.createElement('button');
        btn.textContent = preset.label;
        btn.dataset.presetId = preset.id;
        if (idx === 0) btn.classList.add('active');
        btn.addEventListener('click', () => {var_prefix}ActivatePreset(preset.id));
        container.appendChild(btn);
    }});

    // "Clear all" button
    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'Clear all';
    clearBtn.addEventListener('click', () => {{
        document.querySelectorAll('#{id_prefix}preset-buttons button').forEach(b => b.classList.remove('active'));
        clearBtn.classList.add('active');
        {var_prefix}Visible.clear();
        document.querySelectorAll('#{id_prefix}run-legend label').forEach(label => {{
            const cb = label.querySelector('input');
            cb.checked = false;
            label.classList.add('unchecked');
        }});
        const explEl = document.getElementById('{id_prefix}preset-explanation');
        if (explEl) explEl.textContent = '';
        {var_prefix}Redraw();
    }});
    container.appendChild(clearBtn);
}}

function {var_prefix}BuildLegend() {{
    const container = document.getElementById('{id_prefix}run-legend');
    if (!container) return;

    {var_prefix}Data.runs.forEach(run => {{
        const label = document.createElement('label');
        label.dataset.runId = run.id;

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = true;
        cb.dataset.runId = run.id;
        cb.addEventListener('change', () => {{
            if (cb.checked) {{
                {var_prefix}Visible.add(run.id);
                label.classList.remove('unchecked');
            }} else {{
                {var_prefix}Visible.delete(run.id);
                label.classList.add('unchecked');
            }}
            document.querySelectorAll('#{id_prefix}preset-buttons button').forEach(b => b.classList.remove('active'));
            {var_prefix}Redraw();
        }});

        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';
        swatch.style.background = run.color;

        const text = document.createTextNode(run.label);

        label.appendChild(cb);
        label.appendChild(swatch);
        label.appendChild(text);
        container.appendChild(label);
    }});
}}

function {var_prefix}ActivatePreset(presetId) {{
    const preset = {var_prefix}Presets.find(p => p.id === presetId);
    if (!preset) return;

    document.querySelectorAll('#{id_prefix}preset-buttons button').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.presetId === presetId);
    }});

    {var_prefix}Visible.clear();
    preset.runs.forEach(id => {var_prefix}Visible.add(id));

    document.querySelectorAll('#{id_prefix}run-legend label').forEach(label => {{
        const cb = label.querySelector('input');
        const isVisible = {var_prefix}Visible.has(cb.dataset.runId);
        cb.checked = isVisible;
        label.classList.toggle('unchecked', !isVisible);
    }});

    const explEl = document.getElementById('{id_prefix}preset-explanation');
    if (explEl) explEl.textContent = preset.text || '';

    {var_prefix}Redraw();
}}

// --- Plot configs ---

const {var_prefix}PlotConfigs = [
    {{ id: '{id_prefix}plot-reward', title: 'Reward', yKey: 'reward_overall', format: '.3f' }},
    {{ id: '{id_prefix}plot-completion-len', title: 'Completion Length', yKey: 'completion_len', format: '.0f' }},
    {{ id: '{id_prefix}plot-allcorr', title: 'Frac All Correct', yKey: 'frac_all_correct', format: '.1%' }},
    {{ id: '{id_prefix}plot-allwrong', title: 'Frac All Wrong', yKey: 'frac_all_wrong', format: '.1%' }},
    {{ id: '{id_prefix}plot-loss-grpo', title: 'GRPO Loss', yKey: 'loss_grpo', format: '.4f' }},
    {{ id: '{id_prefix}plot-loss-sft', title: 'SFT Loss', yKey: 'loss_sft', format: '.4f' }},
    {{ id: '{id_prefix}plot-kl', title: 'KL Divergence', yKey: 'kl_mean', format: '.4f' }},
    {{ id: '{id_prefix}plot-entropy', title: 'Entropy', yKey: 'entropy_mean', format: '.3f' }},
    {{ id: '{id_prefix}plot-eval-gsm8k', title: 'GSM8K Accuracy', stepsKey: 'eval_gsm8k_steps',
       yKey: 'eval_gsm8k_accuracy', format: '.1%', linear: true }},
    {{ id: '{id_prefix}plot-eval-math', title: 'MATH Accuracy', stepsKey: 'eval_math_steps',
       yKey: 'eval_math_accuracy', format: '.1%', linear: true }},
    {{ id: '{id_prefix}plot-eval-ifeval', title: 'IFEval Accuracy', stepsKey: 'eval_ifeval_steps',
       yKey: 'eval_ifeval_accuracy', format: '.1%', linear: true }},
    {{ id: '{id_prefix}plot-eval-ifbench_test', title: 'IFBench Accuracy', stepsKey: 'eval_ifbench_test_steps',
       yKey: 'eval_ifbench_test_accuracy', format: '.1%', linear: true }},
];

const {var_prefix}PassKConfigs = {passk_configs_json};

function {var_prefix}UpdateTableHighlight() {{
    const table = document.querySelector('#{id_prefix}comparison-table table');
    if (!table) return;
    table.querySelectorAll('[data-run-id]').forEach(cell => {{
        cell.classList.toggle('col-selected', {var_prefix}Visible.has(cell.dataset.runId));
        cell.classList.toggle('col-dimmed', !{var_prefix}Visible.has(cell.dataset.runId));
    }});
}}

function {var_prefix}Redraw() {{
    {var_prefix}PlotConfigs.forEach(config => {{
        const container = document.getElementById(config.id);
        if (!container) return;
        {var_prefix}DrawPlot(container, {var_prefix}Data.runs, config, {var_prefix}Visible);
    }});
    {var_prefix}PassKConfigs.forEach(config => {{
        const container = document.getElementById(config.id);
        if (!container) return;
        {var_prefix}DrawPassKPlot(container, {var_prefix}Data.runs, config, {var_prefix}Visible);
    }});
    {var_prefix}UpdateTableHighlight();
}}

function {var_prefix}DrawPlot(container, runs, config, highlightedSet) {{
    container.innerHTML = '';

    const margin = {{ top: 16, right: 12, bottom: 28, left: 44 }};
    const width = 336 - margin.left - margin.right;
    const height = 192 - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('viewBox', `0 0 ${{width + margin.left + margin.right}} ${{height + margin.top + margin.bottom}}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    const g = svg.append('g')
        .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

    const stepsKey = config.stepsKey || 'steps';

    let baseAnchor = null;
    if (config.stepsKey && {var_prefix}Data.base_model) {{
        const benchName = config.stepsKey.replace('eval_', '').replace('_steps', '');
        const baseAcc = {var_prefix}Data.base_model.benchmarks[benchName];
        if (baseAcc !== undefined) baseAnchor = baseAcc;
    }}

    // Compute scales from ALL runs (not just highlighted)
    let allX = [];
    let allY = [];

    runs.forEach(run => {{
        const xArr = run[stepsKey];
        if (!xArr || xArr.length === 0) return;
        if (baseAnchor !== null) {{ allX.push(0); allY.push(baseAnchor); }}
        allX.push(...xArr);
        if (config.multiLine) {{
            config.yKeys.forEach(key => {{
                allY.push(...(run[key] || []).filter(v => v !== null && v !== undefined));
            }});
        }} else {{
            allY.push(...(run[config.yKey] || []).filter(v => v !== null && v !== undefined));
        }}
    }});

    if (allY.length === 0) return;

    const xMax = 200;
    allX = allX.filter(v => v <= xMax);
    allY = [];
    runs.forEach(run => {{
        const xArr = run[stepsKey];
        if (!xArr || xArr.length === 0) return;
        if (config.multiLine) {{
            config.yKeys.forEach(key => {{
                const yArr = run[key] || [];
                xArr.forEach((x, i) => {{ if (x <= xMax && yArr[i] != null) allY.push(yArr[i]); }});
            }});
        }} else {{
            const yArr = run[config.yKey] || [];
            xArr.forEach((x, i) => {{ if (x <= xMax && yArr[i] != null) allY.push(yArr[i]); }});
        }}
        if (baseAnchor !== null) allY.push(baseAnchor);
    }});

    const xScale = d3.scaleLinear().domain([d3.min(allX), xMax]).range([0, width]);

    let yMin = d3.min(allY), yMax_ = d3.max(allY);
    const yPad = (yMax_ - yMin) * 0.1 || 0.1;
    const yScale = config.logScale
        ? d3.scaleLog().domain([d3.min(allY.filter(v => v > 0)) * 0.5, d3.max(allY.filter(v => v > 0)) * 2]).range([height, 0]).nice()
        : d3.scaleLinear().domain([yMin - yPad, yMax_ + yPad]).range([height, 0]).nice();

    g.append('g').attr('class', 'grid').selectAll('line').data(yScale.ticks(4)).enter().append('line')
        .attr('x1', 0).attr('x2', width).attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
        .attr('stroke', '#e5e7eb').attr('stroke-width', 1);

    g.append('g').attr('transform', `translate(0,${{height}})`)
        .call(d3.axisBottom(xScale).ticks(4).tickSize(0).tickPadding(6))
        .call(g => g.select('.domain').attr('stroke', '#d1d5db'))
        .call(g => g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px').attr('font-family', 'Source Code Pro, monospace'));

    const yAxis = config.logScale
        ? d3.axisLeft(yScale).ticks(3, '.0e').tickSize(0).tickPadding(6)
        : d3.axisLeft(yScale).ticks(4).tickSize(0).tickPadding(6);
    g.append('g').call(yAxis)
        .call(g => g.select('.domain').remove())
        .call(g => g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px').attr('font-family', 'Source Code Pro, monospace'));

    svg.append('text').attr('x', margin.left + width / 2).attr('y', 14)
        .attr('text-anchor', 'middle').attr('fill', '#1f2937').attr('font-size', '10px')
        .attr('font-weight', '600').attr('font-family', 'Lora, serif').text(config.title);

    const line = d3.line()
        .defined(d => d.y != null && (!config.logScale || d.y > 0))
        .x(d => xScale(d.x)).y(d => yScale(d.y))
        .curve(config.linear ? d3.curveLinear : d3.curveMonotoneX);

    const allLineData = [];
    const GRAY = '#d1d5db';

    // Draw in two passes: gray (non-highlighted) first, then colored (highlighted) on top
    const drawPass = (runSubset, isHighlighted) => {{
        runSubset.forEach((run, runIdx) => {{
            const xArr = run[stepsKey];
            if (!xArr || xArr.length === 0) return;
            const drawColor = isHighlighted ? run.color : GRAY;
            const drawWidth = isHighlighted ? 1.75 : 1;
            const drawOpacity = isHighlighted ? 0.9 : 0.35;

            if (config.multiLine) {{
                config.yKeys.forEach((yKey, keyIdx) => {{
                    const data = xArr.map((step, i) => ({{ x: step, y: (run[yKey] || [])[i], label: run.label, metric: config.yLabels[keyIdx] }}))
                        .filter(d => d.x <= xMax && d.y != null);
                    if (isHighlighted) allLineData.push({{ data, color: run.color, label: run.label, metric: config.yLabels[keyIdx] }});
                    g.append('path').datum(data).attr('fill', 'none').attr('stroke', drawColor)
                        .attr('stroke-width', drawWidth).attr('stroke-opacity', isHighlighted ? (keyIdx === 0 ? 0.9 : 0.5) : drawOpacity)
                        .attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round').attr('d', line);
                }});
            }} else {{
                let data = xArr.map((step, i) => ({{ x: step, y: (run[config.yKey] || [])[i], label: run.label }}))
                    .filter(d => d.x <= xMax && d.y != null && (!config.logScale || d.y > 0));
                if (baseAnchor !== null && data.length > 0) {{
                    data = [{{ x: 0, y: baseAnchor, label: 'Base model' }}, ...data];
                }}
                if (isHighlighted) allLineData.push({{ data, color: run.color, label: run.label }});
                g.append('path').datum(data).attr('fill', 'none').attr('stroke', drawColor)
                    .attr('stroke-width', drawWidth).attr('stroke-opacity', drawOpacity)
                    .attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round').attr('d', line);
            }}
        }});
    }};

    // Pass 1: non-highlighted (gray background)
    drawPass(runs.filter(r => !highlightedSet.has(r.id)), false);
    // Pass 2: highlighted (colored, on top)
    drawPass(runs.filter(r => highlightedSet.has(r.id)), true);

    // Tooltip interaction
    const tt = {var_prefix}GetTooltip();
    const fmt = d3.format(config.format || '.3f');

    const hoverLine = g.append('line').attr('stroke', '#9ca3af').attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3').attr('y1', 0).attr('y2', height).style('opacity', 0);
    const hoverDots = g.append('g').attr('class', 'hover-dots');

    g.append('rect').attr('width', width).attr('height', height).attr('fill', 'transparent')
        .on('mousemove', function(event) {{
            const [mx] = d3.pointer(event);
            const x0 = xScale.invert(mx);
            const bisect = d3.bisector(d => d.x).left;
            let html = `<strong>Step ${{Math.round(x0)}}</strong><br>`;
            hoverLine.attr('x1', mx).attr('x2', mx).style('opacity', 1);
            hoverDots.selectAll('circle').remove();
            allLineData.forEach(li => {{
                const i = bisect(li.data, x0, 1);
                const d0 = li.data[i - 1], d1 = li.data[i];
                if (!d0 && !d1) return;
                const d = !d1 ? d0 : !d0 ? d1 : (x0 - d0.x > d1.x - x0 ? d1 : d0);
                if (d && d.y != null) {{
                    const ml = li.metric ? ` (${{li.metric}})` : '';
                    html += `<span style="color:${{li.color}}">●</span> ${{li.label}}${{ml}}: ${{fmt(d.y)}}<br>`;
                    hoverDots.append('circle').attr('cx', xScale(d.x)).attr('cy', yScale(d.y))
                        .attr('r', 4).attr('fill', li.color).attr('stroke', '#fff').attr('stroke-width', 1.5);
                }}
            }});
            tt.html(html).style('left', (event.pageX + 15) + 'px').style('top', (event.pageY - 10) + 'px').style('opacity', 1);
        }})
        .on('mouseleave', function() {{
            hoverLine.style('opacity', 0);
            hoverDots.selectAll('circle').remove();
            tt.style('opacity', 0);
        }});
}}

// --- Pass@k plot drawing ---
function {var_prefix}DrawPassKPlot(container, runs, config, highlightedSet) {{
    container.innerHTML = '';

    const margin = {{ top: 16, right: 12, bottom: 28, left: 44 }};
    const width = 336 - margin.left - margin.right;
    const height = 192 - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('viewBox', `0 0 ${{width + margin.left + margin.right}} ${{height + margin.top + margin.bottom}}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    const g = svg.append('g')
        .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

    // Collect all k values and rates (including base model)
    let allK = [];
    let allY = [];
    // Extract benchmark key from config (e.g. "passk_gsm8k_pass128_k" -> "gsm8k_pass128")
    const benchKey = config.kKey.replace('passk_', '').replace('_k', '');
    const basePassk = {var_prefix}Data.base_model && {var_prefix}Data.base_model.passk && {var_prefix}Data.base_model.passk[benchKey];
    if (basePassk) {{
        allK.push(...basePassk.k);
        allY.push(...basePassk.rate);
    }}
    runs.forEach(run => {{
        const kArr = run[config.kKey];
        const rArr = run[config.rateKey];
        if (!kArr || !rArr) return;
        allK.push(...kArr);
        allY.push(...rArr.filter(v => v != null));
    }});

    if (allY.length === 0) return;

    const kMin = d3.min(allK);
    const kMax = d3.max(allK);
    const xScale = d3.scaleLog().base(2).domain([kMin, kMax]).range([0, width]).nice();
    const yMin = d3.min(allY), yMax_ = d3.max(allY);
    const yPad = (yMax_ - yMin) * 0.1 || 0.01;
    const yScale = d3.scaleLinear().domain([Math.max(0, yMin - yPad), Math.min(1, yMax_ + yPad)]).range([height, 0]).nice();

    // Grid
    g.append('g').attr('class', 'grid').selectAll('line').data(yScale.ticks(4)).enter().append('line')
        .attr('x1', 0).attr('x2', width).attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
        .attr('stroke', '#e5e7eb').attr('stroke-width', 1);

    // X axis (log2 ticks: 1, 2, 4, 8, 16, 32, 64, 128)
    const tickVals = [1, 2, 4, 8, 16, 32, 64, 128].filter(v => v >= kMin && v <= kMax);
    g.append('g').attr('transform', `translate(0,${{height}})`)
        .call(d3.axisBottom(xScale).tickValues(tickVals).tickFormat(d => d).tickSize(0).tickPadding(6))
        .call(g => g.select('.domain').attr('stroke', '#d1d5db'))
        .call(g => g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px').attr('font-family', 'Source Code Pro, monospace'));

    // Y axis
    g.append('g').call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.0%')).tickSize(0).tickPadding(6))
        .call(g => g.select('.domain').remove())
        .call(g => g.selectAll('.tick text').attr('fill', '#6b7280').attr('font-size', '10px').attr('font-family', 'Source Code Pro, monospace'));

    // Title
    svg.append('text').attr('x', margin.left + width / 2).attr('y', 14)
        .attr('text-anchor', 'middle').attr('fill', '#1f2937').attr('font-size', '10px')
        .attr('font-weight', '600').attr('font-family', 'Lora, serif').text(config.title);

    // X-axis label
    svg.append('text').attr('x', margin.left + width / 2).attr('y', height + margin.top + 28)
        .attr('text-anchor', 'middle').attr('fill', '#9ca3af').attr('font-size', '9px')
        .attr('font-family', 'Source Code Pro, monospace').text('k');

    const line = d3.line()
        .defined(d => d.y != null)
        .x(d => xScale(d.x)).y(d => yScale(d.y))
        .curve(d3.curveMonotoneX);

    const GRAY = '#d1d5db';
    const allLineData = [];

    const drawPass = (runSubset, isHighlighted) => {{
        runSubset.forEach(run => {{
            const kArr = run[config.kKey];
            const rArr = run[config.rateKey];
            if (!kArr || !rArr) return;
            const data = kArr.map((k, i) => ({{ x: k, y: rArr[i], label: run.label }}));
            const drawColor = isHighlighted ? run.color : GRAY;
            const drawWidth = isHighlighted ? 1.75 : 1;
            const drawOpacity = isHighlighted ? 0.9 : 0.35;
            if (isHighlighted) allLineData.push({{ data, color: run.color, label: run.label }});
            g.append('path').datum(data).attr('fill', 'none').attr('stroke', drawColor)
                .attr('stroke-width', drawWidth).attr('stroke-opacity', drawOpacity)
                .attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round').attr('d', line);
        }});
    }};

    drawPass(runs.filter(r => !highlightedSet.has(r.id)), false);
    drawPass(runs.filter(r => highlightedSet.has(r.id)), true);

    // Draw base model line (dashed)
    if (basePassk) {{
        const baseData = basePassk.k.map((k, i) => ({{ x: k, y: basePassk.rate[i], label: {var_prefix}Data.base_model.label }}));
        // Filter to same k range as trained models
        const filteredBase = baseData.filter(d => d.x >= kMin && d.x <= kMax);
        g.append('path').datum(filteredBase).attr('fill', 'none').attr('stroke', '#999')
            .attr('stroke-width', 1.5).attr('stroke-opacity', 0.7)
            .attr('stroke-dasharray', '4,3')
            .attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round').attr('d', line);
        allLineData.push({{ data: filteredBase, color: '#999', label: {var_prefix}Data.base_model.label }});
    }}

    // Tooltip
    const tt = {var_prefix}GetTooltip();
    const fmt = d3.format(config.format || '.1%');

    const hoverLine = g.append('line').attr('stroke', '#9ca3af').attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3').attr('y1', 0).attr('y2', height).style('opacity', 0);
    const hoverDots = g.append('g').attr('class', 'hover-dots');

    g.append('rect').attr('width', width).attr('height', height).attr('fill', 'transparent')
        .on('mousemove', function(event) {{
            const [mx] = d3.pointer(event);
            const x0 = xScale.invert(mx);
            const bisect = d3.bisector(d => d.x).left;
            let html = `<strong>k=${{Math.round(x0)}}</strong><br>`;
            hoverLine.attr('x1', mx).attr('x2', mx).style('opacity', 1);
            hoverDots.selectAll('circle').remove();
            allLineData.forEach(li => {{
                const i = bisect(li.data, x0, 1);
                const d0 = li.data[i - 1], d1 = li.data[i];
                if (!d0 && !d1) return;
                const d = !d1 ? d0 : !d0 ? d1 : (x0 - d0.x > d1.x - x0 ? d1 : d0);
                if (d && d.y != null) {{
                    html += `<span style="color:${{li.color}}">●</span> ${{li.label}}: ${{fmt(d.y)}}<br>`;
                    hoverDots.append('circle').attr('cx', xScale(d.x)).attr('cy', yScale(d.y))
                        .attr('r', 4).attr('fill', li.color).attr('stroke', '#fff').attr('stroke-width', 1.5);
                }}
            }});
            tt.html(html).style('left', (event.pageX + 15) + 'px').style('top', (event.pageY - 10) + 'px').style('opacity', 1);
        }})
        .on('mouseleave', function() {{
            hoverLine.style('opacity', 0);
            hoverDots.selectAll('circle').remove();
            tt.style('opacity', 0);
        }});
}}

// Initialize
function {var_prefix}Init() {{
    {var_prefix}BuildPresets();
    {var_prefix}BuildLegend();
    if ({var_prefix}Presets.length > 0) {{
        {var_prefix}ActivatePreset({var_prefix}Presets[0].id);
    }} else {{
        {var_prefix}Redraw();
    }}
    {var_prefix}PopulateMetrics();
    {var_prefix}GenerateTable();
}}

if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', {var_prefix}Init);
}} else {{
    {var_prefix}Init();
}}
"""
    return js_code


def main():
    parser = argparse.ArgumentParser(description="Generate experiment comparison plots")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    plot_data = generate_plot_data(config)
    presets = config.get("presets", [])
    html_prefix = config.get("html_prefix", "")
    js_code = generate_js(plot_data, presets, html_prefix)

    output_path = config.get("output", "plots.js")
    with open(output_path, "w") as f:
        f.write(js_code)

    print(f"Generated {output_path} (prefix={html_prefix or 'none'})", file=sys.stderr)


if __name__ == "__main__":
    main()
