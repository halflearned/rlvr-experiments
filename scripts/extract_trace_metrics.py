#!/usr/bin/env python3
"""Extract key metrics from a trace.jsonl file.

Usage:
    python scripts/extract_trace_metrics.py <trace_file> [--csv] [--last N]

Extracts per-step:
    - step
    - reward_overall, reward_used, frac_all_correct, frac_all_wrong
    - kl_mean, entropy_mean
    - loss
    - mfu
    - completion_len (mean)
"""

import argparse
import json
import sys
from pathlib import Path


def extract_metrics(trace_path: str, last_n: int | None = None) -> list[dict]:
    """Extract metrics from trace file, returning list of per-step dicts."""

    # Collect events by approximate timestamp groups
    # We'll match events that occur close together as belonging to same step

    reward_stats_events = []
    grpo_debug_events = []
    metrics_events = []
    titan_metrics_events = []
    batch_padding_events = []

    with open(trace_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('type') != 'counter':
                continue
            name = obj.get('name')
            if name == 'reward_stats':
                reward_stats_events.append(obj)
            elif name == 'grpo.debug':
                grpo_debug_events.append(obj)
            elif name == 'metrics':
                metrics_events.append(obj)
            elif name == 'titan.metrics':
                titan_metrics_events.append(obj)
            elif name == 'batch.padding':
                batch_padding_events.append(obj)

    # Build per-step records
    # reward_stats has step info implicitly (it's emitted once per optim step)
    # batch.padding has explicit 'step' field

    # Use batch.padding as the anchor since it has step numbers
    step_to_padding = {e['step']: e for e in batch_padding_events if 'step' in e}

    # Match other events by timestamp proximity to batch.padding
    def find_nearest(events: list[dict], target_ts: float, tolerance: float = 5.0):
        """Find event with timestamp closest to target, within tolerance."""
        best = None
        best_dist = float('inf')
        for e in events:
            dist = abs(e['ts'] - target_ts)
            if dist < best_dist and dist < tolerance:
                best = e
                best_dist = dist
        return best

    results = []
    steps = sorted(step_to_padding.keys())

    for step in steps:
        padding = step_to_padding[step]
        ts = padding['ts']

        record = {'step': step}

        # Find matching events
        reward_stat = find_nearest(reward_stats_events, ts)
        if reward_stat:
            record['reward_overall'] = reward_stat.get('reward_overall')
            record['reward_used'] = reward_stat.get('reward_used')
            record['frac_all_correct'] = reward_stat.get('frac_all_correct')
            record['frac_all_wrong'] = reward_stat.get('frac_all_wrong')

        grpo_debug = find_nearest(grpo_debug_events, ts)
        if grpo_debug:
            record['kl_mean'] = grpo_debug.get('kl_mean')
            record['entropy_mean'] = grpo_debug.get('entropy_mean')

        metrics = find_nearest(metrics_events, ts)
        if metrics:
            record['loss'] = metrics.get('loss')

        titan = find_nearest(titan_metrics_events, ts, tolerance=60.0)  # titan emits less frequently
        if titan:
            record['mfu'] = titan.get('mfu')

        # Compute mean completion length from batch.padding
        comp_lens = padding.get('completion_lens', [])
        if comp_lens:
            record['completion_len'] = sum(comp_lens) / len(comp_lens)

        results.append(record)

    if last_n is not None:
        results = results[-last_n:]

    return results


def print_table(results: list[dict]):
    """Print results as a formatted table."""
    if not results:
        print("No data found.")
        return

    # Column definitions: (key, header, format, width)
    columns = [
        ('step', 'Step', 'd', 6),
        ('reward_overall', 'Reward', '.3f', 8),
        ('reward_used', 'RewUsed', '.3f', 8),
        ('frac_all_correct', 'AllCorr', '.3f', 8),
        ('frac_all_wrong', 'AllWrong', '.3f', 8),
        ('kl_mean', 'KL', '.2e', 9),
        ('entropy_mean', 'Entropy', '.3f', 8),
        ('loss', 'Loss', '.4f', 8),
        ('mfu', 'MFU%', '.1f', 6),
        ('completion_len', 'CompLen', '.0f', 8),
    ]

    # Print header
    header = ' | '.join(f"{col[1]:>{col[3]}}" for col in columns)
    print(header)
    print('-' * len(header))

    # Print rows
    for row in results:
        cells = []
        for key, _, fmt, width in columns:
            val = row.get(key)
            if val is None:
                cells.append(' ' * width)
            else:
                cells.append(f"{val:{fmt}}".rjust(width))
        print(' | '.join(cells))


def print_csv(results: list[dict]):
    """Print results as CSV."""
    if not results:
        return

    columns = ['step', 'reward_overall', 'reward_used', 'frac_all_correct',
               'frac_all_wrong', 'kl_mean', 'entropy_mean', 'loss', 'mfu', 'completion_len']

    print(','.join(columns))
    for row in results:
        cells = [str(row.get(c, '')) for c in columns]
        print(','.join(cells))


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from trace file')
    parser.add_argument('trace_file', help='Path to trace.jsonl file')
    parser.add_argument('--csv', action='store_true', help='Output as CSV')
    parser.add_argument('--last', type=int, metavar='N', help='Only show last N steps')
    args = parser.parse_args()

    if not Path(args.trace_file).exists():
        print(f"Error: File not found: {args.trace_file}", file=sys.stderr)
        sys.exit(1)

    results = extract_metrics(args.trace_file, last_n=args.last)

    if args.csv:
        print_csv(results)
    else:
        print_table(results)


if __name__ == '__main__':
    main()
