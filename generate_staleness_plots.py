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
import json
import sys
from pathlib import Path

import yaml


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
        metrics, raw_events = extract_metrics(str(trace_path))

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
                    summary_jsonl = bench_dir / "summary.jsonl"
                    summary_json = bench_dir / "summary.json"
                    if summary_jsonl.exists():
                        step_to_acc = {}
                        with open(summary_jsonl) as f:
                            for line in f:
                                entry = json.loads(line)
                                if "step" in entry and "accuracy" in entry:
                                    step_to_acc[entry["step"]] = entry["accuracy"]
                                    if entry["step"] == 200:
                                        summaries[run_id]["benchmarks"][bench_name] = {
                                            "accuracy": entry["accuracy"],
                                            "step": 200,
                                        }
                        if step_to_acc:
                            entries = sorted(step_to_acc.items())
                            eval_curves[bench_name] = entries
                    elif summary_json.exists():
                        with open(summary_json) as f:
                            entry = json.load(f)
                        if "accuracy" in entry:
                            summaries[run_id]["benchmarks"][bench_name] = {
                                "accuracy": entry.get("accuracy"),
                                "step": entry.get("step"),
                            }

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
        for bench_name, entries in eval_curves.items():
            run_entry[f"eval_{bench_name}_steps"] = [s for s, _ in entries]
            run_entry[f"eval_{bench_name}_accuracy"] = [a for _, a in entries]

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

    result = {"runs": runs_data, "summaries": summaries}
    if base_benchmarks:
        result["base_model"] = {
            "label": base_config.get("label", "Base"),
            "benchmarks": base_benchmarks,
        }
    return result


def generate_js(plot_data: dict, presets: list[dict] | None = None) -> str:
    """Generate D3.js code for the plots."""
    data_json = json.dumps(plot_data, indent=2)
    presets_json = json.dumps(presets or [], indent=2)

    js_code = f"""// Auto-generated experiment comparison plots
// Generated by generate_staleness_plots.py

const stalenessData = {data_json};
const presetConfigs = {presets_json};

// Format specifications for different metric types
const metricFormats = {{
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

// Populate inline metric spans
function populateInlineMetrics() {{
    const spans = document.querySelectorAll('span.metric[data-run][data-key]');
    spans.forEach(span => {{
        const runId = span.dataset.run;
        const key = span.dataset.key;
        const summary = stalenessData.summaries[runId];
        if (!summary) {{
            console.warn(`No summary found for run: ${{runId}}`);
            span.textContent = '??';
            return;
        }}
        const value = summary[key];
        if (value === null || value === undefined) {{
            span.textContent = 'N/A';
            return;
        }}
        const fmt = metricFormats[key] || {{ format: '.3f', suffix: '' }};
        const formatted = d3.format(fmt.format)(value) + fmt.suffix;
        span.textContent = formatted;
        // Optionally color-code by run
        if (summary.color) {{
            span.style.color = summary.color;
            span.style.fontWeight = '600';
        }}
    }});
}}

// Generate comparison table
function generateComparisonTable() {{
    const container = document.getElementById('staleness-comparison-table');
    if (!container) return;

    const runs = stalenessData.runs;
    const summaries = stalenessData.summaries;
    const baseModel = stalenessData.base_model || null;

    // Collect all benchmark names across all runs
    const benchSet = new Set();
    runs.forEach(run => {{
        const s = summaries[run.id];
        if (s && s.benchmarks) Object.keys(s.benchmarks).forEach(b => benchSet.add(b));
    }});
    if (baseModel) Object.keys(baseModel.benchmarks).forEach(b => benchSet.add(b));
    const benchmarks = Array.from(benchSet).sort();

    if (benchmarks.length === 0) return;

    const ncols = runs.length + (baseModel ? 2 : 1); // +1 for row label, +1 for base

    let html = '<table class="results-table comparison-table">';

    // Header
    html += '<thead><tr><th>Benchmark (step 200)</th>';
    if (baseModel) html += `<th style="color:#999">${{baseModel.label}}</th>`;
    runs.forEach(run => {{
        html += `<th style="color: ${{run.color}}">${{run.label}}</th>`;
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
                html += `<td>${{d3.format('.1%')(bdata.accuracy)}}</td>`;
            }} else {{
                html += '<td>&mdash;</td>';
            }}
        }});
        html += '</tr>';
    }});

    html += '</tbody></table>';
    container.innerHTML = html;
}}

// Create shared tooltip
let tooltip = null;
function getTooltip() {{
    if (!tooltip) {{
        tooltip = d3.select('body').append('div')
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
    return tooltip;
}}

// --- Preset & checkbox UI ---

// Track which runs are visible
const visibleRuns = new Set(stalenessData.runs.map(r => r.id));

function buildPresetButtons() {{
    const container = document.getElementById('preset-buttons');
    if (!container || presetConfigs.length === 0) return;

    presetConfigs.forEach((preset, idx) => {{
        const btn = document.createElement('button');
        btn.textContent = preset.label;
        btn.dataset.presetId = preset.id;
        if (idx === 0) btn.classList.add('active');
        btn.addEventListener('click', () => activatePreset(preset.id));
        container.appendChild(btn);
    }});
}}

function buildRunLegend() {{
    const container = document.getElementById('run-legend');
    if (!container) return;

    stalenessData.runs.forEach(run => {{
        const label = document.createElement('label');
        label.dataset.runId = run.id;

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = true;
        cb.dataset.runId = run.id;
        cb.addEventListener('change', () => {{
            if (cb.checked) {{
                visibleRuns.add(run.id);
                label.classList.remove('unchecked');
            }} else {{
                visibleRuns.delete(run.id);
                label.classList.add('unchecked');
            }}
            // Deactivate preset buttons since user is manually toggling
            document.querySelectorAll('#preset-buttons button').forEach(b => b.classList.remove('active'));
            redrawPlots();
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

function activatePreset(presetId) {{
    const preset = presetConfigs.find(p => p.id === presetId);
    if (!preset) return;

    // Update button active state
    document.querySelectorAll('#preset-buttons button').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.presetId === presetId);
    }});

    // Update checkboxes
    visibleRuns.clear();
    preset.runs.forEach(id => visibleRuns.add(id));

    document.querySelectorAll('#run-legend label').forEach(label => {{
        const cb = label.querySelector('input');
        const isVisible = visibleRuns.has(cb.dataset.runId);
        cb.checked = isVisible;
        label.classList.toggle('unchecked', !isVisible);
    }});

    // Update explanation text
    const explEl = document.getElementById('preset-explanation');
    if (explEl) explEl.textContent = preset.text || '';

    redrawPlots();
}}

// --- Plot configs (static) ---

const plotConfigs = [
    // Row 1: Rewards
    {{
        id: 'plot-reward',
        title: 'Reward',
        yKey: 'reward_overall',
        yLabel: 'Reward',
        logScale: false,
        format: '.3f'
    }},
    {{
        id: 'plot-allcorr-allwrong',
        title: 'All Correct / All Wrong',
        yKeys: ['frac_all_correct', 'frac_all_wrong'],
        yLabels: ['Correct', 'Wrong'],
        yLabel: 'Fraction',
        logScale: false,
        multiLine: true,
        format: '.1%'
    }},
    {{
        id: 'plot-completion-len',
        title: 'Completion Length',
        yKey: 'completion_len',
        yLabel: 'Tokens',
        logScale: false,
        format: '.0f'
    }},
    // Row 2: Training dynamics
    {{
        id: 'plot-loss-grpo',
        title: 'GRPO Loss',
        yKey: 'loss_grpo',
        yLabel: 'Loss',
        logScale: false,
        format: '.4f'
    }},
    {{
        id: 'plot-loss-sft',
        title: 'SFT Loss',
        yKey: 'loss_sft',
        yLabel: 'Loss',
        logScale: false,
        format: '.4f'
    }},
    {{
        id: 'plot-kl',
        title: 'KL Divergence',
        yKey: 'kl_mean',
        yLabel: 'KL',
        logScale: true,
        format: '.2e'
    }},
    {{
        id: 'plot-entropy',
        title: 'Entropy',
        yKey: 'entropy_mean',
        yLabel: 'Entropy',
        logScale: false,
        format: '.3f'
    }},
    // Row 3: Evaluation
    {{
        id: 'plot-eval-gsm8k',
        title: 'GSM8K Accuracy',
        stepsKey: 'eval_gsm8k_steps',
        yKey: 'eval_gsm8k_accuracy',
        yLabel: 'Accuracy',
        logScale: false,
        format: '.1%',
        linear: true
    }},
    {{
        id: 'plot-eval-math',
        title: 'MATH Accuracy',
        stepsKey: 'eval_math_steps',
        yKey: 'eval_math_accuracy',
        yLabel: 'Accuracy',
        logScale: false,
        format: '.1%',
        linear: true
    }}
];

function redrawPlots() {{
    const activeRuns = stalenessData.runs.filter(r => visibleRuns.has(r.id));
    plotConfigs.forEach(config => {{
        const container = document.getElementById(config.id);
        if (!container) return;
        drawD3Plot(container, activeRuns, config);
    }});
}}

function initStalenessPlots() {{
    console.log('initStalenessPlots called, D3 version:', d3.version);
    buildPresetButtons();
    buildRunLegend();

    // Activate first preset if available, otherwise show all
    if (presetConfigs.length > 0) {{
        activatePreset(presetConfigs[0].id);
    }} else {{
        redrawPlots();
    }}
}}

function drawD3Plot(container, runs, config) {{
    // Clear any existing content
    container.innerHTML = '';

    const margin = {{ top: 24, right: 12, bottom: 32, left: 48 }};
    const width = 280 - margin.left - margin.right;
    const height = 170 - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('viewBox', `0 0 ${{width + margin.left + margin.right}} ${{height + margin.top + margin.bottom}}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    const g = svg.append('g')
        .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

    // Collect all data points
    // stepsKey allows eval plots to use per-run step arrays (e.g. eval_gsm8k_steps)
    const stepsKey = config.stepsKey || 'steps';
    let allX = [];
    let allY = [];

    runs.forEach(run => {{
        const xArr = run[stepsKey];
        if (!xArr || xArr.length === 0) return;  // skip runs without this data
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

    // X scale
    const xScale = d3.scaleLinear()
        .domain(d3.extent(allX))
        .range([0, width]);

    // Y scale
    let yMin = d3.min(allY);
    let yMax = d3.max(allY);
    const yPad = (yMax - yMin) * 0.1 || 0.1;

    let yScale;
    if (config.logScale) {{
        const posValues = allY.filter(v => v > 0);
        yScale = d3.scaleLog()
            .domain([d3.min(posValues) * 0.5, d3.max(posValues) * 2])
            .range([height, 0])
            .nice();
    }} else {{
        yScale = d3.scaleLinear()
            .domain([yMin - yPad, yMax + yPad])
            .range([height, 0])
            .nice();
    }}

    // Grid lines (horizontal only, subtle)
    g.append('g')
        .attr('class', 'grid')
        .selectAll('line')
        .data(yScale.ticks(4))
        .enter()
        .append('line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', d => yScale(d))
        .attr('y2', d => yScale(d))
        .attr('stroke', '#e5e7eb')
        .attr('stroke-width', 1);

    // X axis
    g.append('g')
        .attr('transform', `translate(0,${{height}})`)
        .call(d3.axisBottom(xScale).ticks(4).tickSize(0).tickPadding(6))
        .call(g => g.select('.domain').attr('stroke', '#d1d5db'))
        .call(g => g.selectAll('.tick text')
            .attr('fill', '#6b7280')
            .attr('font-size', '10px')
            .attr('font-family', 'Source Code Pro, monospace'));

    // Y axis
    const yAxis = config.logScale
        ? d3.axisLeft(yScale).ticks(3, '.0e').tickSize(0).tickPadding(6)
        : d3.axisLeft(yScale).ticks(4).tickSize(0).tickPadding(6);

    g.append('g')
        .call(yAxis)
        .call(g => g.select('.domain').remove())
        .call(g => g.selectAll('.tick text')
            .attr('fill', '#6b7280')
            .attr('font-size', '10px')
            .attr('font-family', 'Source Code Pro, monospace'));

    // Title
    svg.append('text')
        .attr('x', margin.left + width / 2)
        .attr('y', 14)
        .attr('text-anchor', 'middle')
        .attr('fill', '#1f2937')
        .attr('font-size', '12px')
        .attr('font-weight', '600')
        .attr('font-family', 'Lora, serif')
        .text(config.title);

    // Line generator
    const line = d3.line()
        .defined(d => d.y !== null && d.y !== undefined && (!config.logScale || d.y > 0))
        .x(d => xScale(d.x))
        .y(d => yScale(d.y))
        .curve(config.linear ? d3.curveLinear : d3.curveMonotoneX);

    // Colors and styles for runs
    const lineWidths = [1.75, 1.75];
    const dashPatterns = ['', '6,4'];
    const opacities = [0.9, 0.9];

    // Store all line data for tooltip
    const allLineData = [];

    runs.forEach((run, runIdx) => {{
        const xArr = run[stepsKey];
        if (!xArr || xArr.length === 0) return;  // skip runs without this data

        if (config.multiLine) {{
            config.yKeys.forEach((yKey, keyIdx) => {{
                const data = xArr.map((step, i) => ({{
                    x: step,
                    y: (run[yKey] || [])[i],
                    label: run.label,
                    metric: config.yLabels[keyIdx]
                }})).filter(d => d.y !== null && d.y !== undefined);

                allLineData.push({{ data, color: run.color, runIdx, keyIdx, label: run.label, metric: config.yLabels[keyIdx] }});

                g.append('path')
                    .datum(data)
                    .attr('fill', 'none')
                    .attr('stroke', run.color)
                    .attr('stroke-width', lineWidths[runIdx])
                    .attr('stroke-dasharray', dashPatterns[runIdx])
                    .attr('stroke-opacity', keyIdx === 0 ? opacities[runIdx] : 0.5)
                    .attr('stroke-linecap', 'round')
                    .attr('stroke-linejoin', 'round')
                    .attr('d', line);
            }});
        }} else {{
            const data = xArr.map((step, i) => ({{
                x: step,
                y: (run[config.yKey] || [])[i],
                label: run.label
            }})).filter(d => d.y !== null && d.y !== undefined && (!config.logScale || d.y > 0));

            allLineData.push({{ data, color: run.color, runIdx, label: run.label }});

            g.append('path')
                .datum(data)
                .attr('fill', 'none')
                .attr('stroke', run.color)
                .attr('stroke-width', lineWidths[runIdx])
                .attr('stroke-dasharray', dashPatterns[runIdx])
                .attr('stroke-opacity', opacities[runIdx])
                .attr('stroke-linecap', 'round')
                .attr('stroke-linejoin', 'round')
                .attr('d', line);
        }}
    }});

    // Tooltip interaction
    const tt = getTooltip();
    const format = d3.format(config.format || '.3f');

    // Vertical line for hover
    const hoverLine = g.append('line')
        .attr('stroke', '#9ca3af')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .attr('y1', 0)
        .attr('y2', height)
        .style('opacity', 0);

    // Hover dots
    const hoverDots = g.append('g').attr('class', 'hover-dots');

    // Overlay for mouse events
    g.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', 'transparent')
        .on('mousemove', function(event) {{
            const [mx] = d3.pointer(event);
            const x0 = xScale.invert(mx);

            // Find closest step
            const bisect = d3.bisector(d => d.x).left;
            let tooltipHtml = `<strong>Step ${{Math.round(x0)}}</strong><br>`;

            hoverLine.attr('x1', mx).attr('x2', mx).style('opacity', 1);
            hoverDots.selectAll('circle').remove();

            allLineData.forEach(lineInfo => {{
                const i = bisect(lineInfo.data, x0, 1);
                const d0 = lineInfo.data[i - 1];
                const d1 = lineInfo.data[i];
                if (!d0 && !d1) return;
                const d = !d1 ? d0 : !d0 ? d1 : (x0 - d0.x > d1.x - x0 ? d1 : d0);

                if (d && d.y !== null && d.y !== undefined) {{
                    const metricLabel = lineInfo.metric ? ` (${{lineInfo.metric}})` : '';
                    tooltipHtml += `<span style="color:${{lineInfo.color}}">●</span> ${{lineInfo.label}}${{metricLabel}}: ${{format(d.y)}}<br>`;

                    hoverDots.append('circle')
                        .attr('cx', xScale(d.x))
                        .attr('cy', yScale(d.y))
                        .attr('r', 4)
                        .attr('fill', lineInfo.color)
                        .attr('stroke', '#fff')
                        .attr('stroke-width', 1.5);
                }}
            }});

            tt.html(tooltipHtml)
                .style('left', (event.pageX + 15) + 'px')
                .style('top', (event.pageY - 10) + 'px')
                .style('opacity', 1);
        }})
        .on('mouseleave', function() {{
            hoverLine.style('opacity', 0);
            hoverDots.selectAll('circle').remove();
            tt.style('opacity', 0);
        }});
}}

// Initialize when DOM is ready
function initAll() {{
    initStalenessPlots();
    populateInlineMetrics();
    generateComparisonTable();
}}

if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', initAll);
}} else {{
    initAll();
}}
"""
    return js_code


def main():
    parser = argparse.ArgumentParser(description="Generate staleness comparison plots")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    plot_data = generate_plot_data(config)
    presets = config.get("presets", [])
    js_code = generate_js(plot_data, presets)

    output_path = config.get("output", "staleness_plots.js")
    with open(output_path, "w") as f:
        f.write(js_code)

    print(f"Generated {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
