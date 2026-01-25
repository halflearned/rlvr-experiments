#!/usr/bin/env python3
"""Monitor a running training job and alert on KL divergence blowup."""

import json
import sys
import time
from pathlib import Path
from collections import deque

def parse_metrics(trace_path: Path):
    """Parse metrics from trace file."""
    metrics = []
    with open(trace_path) as f:
        for line in f:
            try:
                event = json.loads(line)
                if event.get("name") == "grpo.debug":
                    metrics.append({
                        "ts": event["ts"],
                        "kl_mean": event.get("kl_mean", 0),
                        "kl_max": event.get("kl_max", 0),
                        "entropy_mean": event.get("entropy_mean", 0),
                    })
                elif event.get("name") == "metrics":
                    # Find the matching grpo.debug entry and add metrics info
                    if metrics:
                        metrics[-1].update({
                            "loss": event.get("loss", 0),
                            "grad_norm": event.get("grad_norm", 0),
                            "avg_reward": event.get("avg_reward", 0),
                        })
                elif event.get("name") == "rewards":
                    if metrics:
                        metrics[-1]["reward_mean"] = event.get("mean", 0)
            except json.JSONDecodeError:
                continue
    return metrics

def main():
    if len(sys.argv) < 2:
        # Find latest run
        results_dir = Path("/efs/rlvr-experiments/results")
        runs = sorted(results_dir.glob("qwen3-1.7B-gsm8k_*"))
        if not runs:
            print("No runs found")
            return
        run_dir = runs[-1]
    else:
        run_dir = Path(sys.argv[1])

    trace_path = run_dir / "traces" / "trace.jsonl"
    print(f"Monitoring: {trace_path}")

    kl_threshold = 10.0
    consecutive_high = 0
    last_step = 0

    while True:
        if not trace_path.exists():
            print("Waiting for trace file...")
            time.sleep(5)
            continue

        metrics = parse_metrics(trace_path)

        if len(metrics) > last_step:
            for m in metrics[last_step:]:
                step = len(metrics[:metrics.index(m)+1])
                kl = m.get("kl_max", 0)
                reward = m.get("reward_mean", m.get("avg_reward", 0))
                grad_norm = m.get("grad_norm", 0)

                status = "OK"
                if kl > kl_threshold:
                    consecutive_high += 1
                    status = f"HIGH KL ({consecutive_high} consecutive)"
                else:
                    consecutive_high = 0

                print(f"Step {step}: kl_max={kl:.4f}, reward={reward:.3f}, grad_norm={grad_norm:.3f} [{status}]")

                if consecutive_high >= 3:
                    print("\n" + "="*60)
                    print("ALERT: KL divergence > 10 for 3+ consecutive steps!")
                    print("Recommend: Kill run and adjust params (lower lr or raise beta)")
                    print("="*60)

            last_step = len(metrics)

        time.sleep(10)

if __name__ == "__main__":
    main()
