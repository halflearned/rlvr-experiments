#!/bin/bash
#
# Launch pass@k evaluation across all available GPUs in parallel
#
# Usage:
#   ./scripts/launch_pass_at_k.sh <dataset> <output_dir> [gpus] [extra_args]
#
# Example:
#   ./scripts/launch_pass_at_k.sh gsm8k results/qwen3-1.7B-base/evals/gsm8k/pass-at-k "0,1,2,3,4,5,6,7"
#   ./scripts/launch_pass_at_k.sh if_multi_constraints results/... "0,1,2,3,4,5,6,7" "--shuffle"

set -e

DATASET=${1:-gsm8k}
OUTPUT_DIR=${2:-results/qwen3-1.7B-base/evals/${DATASET}/pass-at-k}
GPUS=${3:-"0,1,2,3,4,5,6,7"}
MAX_MODEL_LEN=${4:-2048}
EXTRA_ARGS=${5:-""}

# Parse GPU list
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "Launching pass@k evaluation"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $GPUS ($NUM_GPUS total)"
echo "Max model len: $MAX_MODEL_LEN"
echo "Extra args: $EXTRA_ARGS"
echo "============================================================"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Launch one process per GPU
PIDS=()
for i in "${!GPU_ARRAY[@]}"; do
    GPU=${GPU_ARRAY[$i]}
    echo "Launching shard $i on GPU $GPU..."

    CUDA_VISIBLE_DEVICES=$GPU /efs/rlvr-experiments/.venv/bin/python \
        /efs/rlvr-experiments/scripts/eval_pass_at_k.py \
        "$DATASET" \
        --split train \
        --n 128 \
        --batch-size 16 \
        --max-tokens 1024 \
        --max-model-len "$MAX_MODEL_LEN" \
        --temperature 1.0 \
        --gpus "$GPU" \
        --gpu-index "$i" \
        --num-shards "$NUM_GPUS" \
        --output-dir "$OUTPUT_DIR" \
        --verifier-workers 8 \
        $EXTRA_ARGS \
        > "$OUTPUT_DIR/shard_${i}.log" 2>&1 &

    PIDS+=($!)
    echo "  Started PID ${PIDS[$i]}"
done

echo ""
echo "All shards launched. Monitoring..."
echo "Logs: $OUTPUT_DIR/shard_*.log"
echo ""

# Wait for all processes
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "Shard $i (PID $PID) completed successfully"
    else
        echo "ERROR: Shard $i (PID $PID) failed with exit code $?"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "ERROR: $FAILED shard(s) failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "All shards completed successfully!"
echo "Merging results..."
echo "============================================================"

# Merge shard results
/efs/rlvr-experiments/.venv/bin/python -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
all_results = []
total_correct = 0
total_completions = 0

for shard_dir in sorted(output_dir.glob('shard_*')):
    results_file = shard_dir / 'verification_results.jsonl'
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    all_results.append(result)
                    total_correct += result['num_correct']
                    total_completions += result['num_completions']

# Compute overall statistics
overall_pass_rate = total_correct / total_completions if total_completions else 0
avg_pass_rate = sum(r['pass_rate'] for r in all_results) / len(all_results) if all_results else 0

# Compute pass@k
n = all_results[0]['num_completions'] if all_results else 0
pass_at_k = {}
for k in [1, 2, 4, 8, 16, 32, 64, 128]:
    if k > n:
        break
    passed = sum(1 for r in all_results if any(s > 0 for s in r['scores'][:k]))
    pass_at_k[f'pass@{k}'] = passed / len(all_results) if all_results else 0

# Write merged results
merged_file = output_dir / 'all_verification_results.jsonl'
with open(merged_file, 'w') as f:
    for r in all_results:
        f.write(json.dumps(r) + '\n')

# Write summary
summary = {
    'dataset': '$DATASET',
    'num_prompts': len(all_results),
    'num_completions': total_completions,
    'num_correct': total_correct,
    'overall_pass_rate': overall_pass_rate,
    'avg_per_prompt_pass_rate': avg_pass_rate,
    'pass_at_k': pass_at_k,
}

summary_file = output_dir / 'merged_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Merged {len(all_results)} prompts from {len(list(output_dir.glob(\"shard_*\")))} shards')
print(f'Results: {merged_file}')
print(f'Summary: {summary_file}')
print()
print('Pass@k:')
for k, rate in pass_at_k.items():
    print(f'  {k}: {rate*100:.2f}%')
print()
print(f'Overall pass rate: {overall_pass_rate*100:.2f}%')
print(f'Avg per-prompt pass rate: {avg_pass_rate*100:.2f}%')
"

echo ""
echo "Done!"
