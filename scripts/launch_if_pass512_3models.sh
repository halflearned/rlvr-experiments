#!/bin/bash
# Launch pass@512 for 3 IF models across 3 nodes
# Each node handles 1 model, running ifeval then ifbench with 8 GPU shards
set -e

SCRIPT="/efs/rlvr-experiments/scripts/eval_pass_at_k.py"
VENV="/efs/rlvr-experiments/.venv/bin/python"

# Model definitions
BASE_MODEL="Qwen/Qwen3-1.7B-Base"
BASE_RESULTS="/efs/rlvr-experiments/results/qwen3-1.7B-base"

GRPO_MODEL="/efs/rlvr-experiments/results/annotations-adhoc-20260201-095517/checkpoints/config_rewritten_20260201-100341_step200"
GRPO_RESULTS="/efs/rlvr-experiments/results/annotations-adhoc-20260201-095517"

DAPO_MODEL="/efs/rlvr-experiments/results/annotations-adhoc-20260130-181815/checkpoints/config_rewritten_20260130-182707_step200"
DAPO_RESULTS="/efs/rlvr-experiments/results/annotations-adhoc-20260130-181815"

run_pass512() {
    local model_path="$1"
    local output_dir="$2"
    local benchmark="$3"
    local gpus="$4"
    local log_prefix="$5"

    IFS=',' read -ra GPU_ARRAY <<< "$gpus"
    local num_gpus=${#GPU_ARRAY[@]}

    echo "[$(date)] Starting $benchmark for $log_prefix ($num_gpus GPUs: $gpus)"

    PIDS=()
    for i in "${!GPU_ARRAY[@]}"; do
        GPU=${GPU_ARRAY[$i]}
        CUDA_VISIBLE_DEVICES=$GPU $VENV -u $SCRIPT \
            "$benchmark" \
            --split train \
            --n 512 \
            --batch-size 8 \
            --max-tokens 2048 \
            --max-model-len 4096 \
            --temperature 1.0 \
            --model-path "$model_path" \
            --gpus "$GPU" \
            --gpu-index "$i" \
            --num-shards "$num_gpus" \
            --output-dir "$output_dir" \
            --verifier-workers 8 \
            > "${output_dir}/shard_${i}.log" 2>&1 &
        PIDS+=($!)
        echo "  Shard $i on GPU $GPU (PID ${PIDS[$i]})"
    done

    # Wait for all
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if ! wait ${PIDS[$i]}; then
            echo "  ERROR: Shard $i failed"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo "  WARNING: $FAILED shards failed for $benchmark/$log_prefix"
        return 1
    fi
    echo "[$(date)] Completed $benchmark for $log_prefix"
    return 0
}

merge_results() {
    local output_dir="$1"
    local benchmark="$2"

    $VENV -u -c "
import json
from pathlib import Path

output_dir = Path('$output_dir')
all_results = []

for shard_dir in sorted(output_dir.glob('shard_*')):
    if not shard_dir.is_dir():
        continue
    results_file = shard_dir / 'verification_results.jsonl'
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

if not all_results:
    print('No results found!')
    import sys; sys.exit(1)

n = all_results[0].get('num_completions', len(all_results[0].get('scores', [])))
pass_at_k = {}
for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    if k > n:
        break
    # Unbiased pass@k estimator
    import math
    def pass_at_k_estimator(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    estimates = []
    for r in all_results:
        c = r.get('num_correct', sum(1 for s in r.get('scores', []) if s > 0))
        estimates.append(pass_at_k_estimator(n, c, k))
    pass_at_k[f'pass@{k}'] = sum(estimates) / len(estimates)

total_correct = sum(r.get('num_correct', sum(1 for s in r.get('scores', []) if s > 0)) for r in all_results)
total_completions = sum(r.get('num_completions', len(r.get('scores', []))) for r in all_results)

summary = {
    'benchmark': '$benchmark',
    'num_prompts': len(all_results),
    'num_completions_per_prompt': n,
    'num_completions': total_completions,
    'num_correct': total_correct,
    'overall_pass_rate': total_correct / total_completions if total_completions else 0,
    'avg_per_prompt_pass_rate': sum(r.get('pass_rate', r.get('num_correct',0)/max(r.get('num_completions',1),1)) for r in all_results) / len(all_results),
    'pass_at_k': pass_at_k,
}

# Write merged verification_results
merged_vr = output_dir / 'verification_results.jsonl'
with open(merged_vr, 'w') as f:
    for r in all_results:
        f.write(json.dumps(r) + '\n')

# Write summary
summary_file = output_dir / 'merged_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Merged {len(all_results)} prompts')
print(f'Pass@k:')
for k, rate in pass_at_k.items():
    print(f'  {k}: {rate*100:.2f}%')
"
}

echo "============================================"
echo "Pass@512 IF evaluation - 3 models, 3 nodes"
echo "============================================"
echo ""

# This script is meant to be sourced or run on each node separately
# Usage: $0 <node_id>  where node_id is 0 (primary), 1 (secondary), 2 (tertiary)
NODE=${1:-0}

case $NODE in
    0)
        echo "=== Primary node: Base model (Qwen3-1.7B-Base) ==="
        MODEL="$BASE_MODEL"
        RESULTS="$BASE_RESULTS"
        LABEL="base"
        GPUS="0,1,2,3,4,5,6,7"
        ;;
    1)
        echo "=== Secondary node: GRPO lr=1e-5 β=1e-4 ==="
        MODEL="$GRPO_MODEL"
        RESULTS="$GRPO_RESULTS"
        LABEL="grpo-lr1e5-beta1e4"
        GPUS="0,1,2,3,4,5,6,7"
        ;;
    2)
        echo "=== Tertiary node: DAPO lr=1e-5 β=1e-4 ==="
        MODEL="$DAPO_MODEL"
        RESULTS="$DAPO_RESULTS"
        LABEL="dapo-lr1e5-beta1e4"
        GPUS="0,1,2,3,4,5,6,7"
        ;;
esac

for BENCH in ifeval ifbench; do
    OUT="${RESULTS}/evals/${BENCH}/pass-at-k-512"
    mkdir -p "$OUT"
    run_pass512 "$MODEL" "$OUT" "$BENCH" "$GPUS" "$LABEL"
    merge_results "$OUT" "$BENCH"
done

echo ""
echo "[$(date)] All done for node $NODE ($LABEL)"
