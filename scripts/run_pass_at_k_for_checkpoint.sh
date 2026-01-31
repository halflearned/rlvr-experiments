#!/bin/bash
# Run pass@k evaluation for a specific checkpoint using all specified GPUs
# Usage: bash scripts/run_pass_at_k_for_checkpoint.sh <checkpoint_path> <dataset> <output_dir> <n_completions> <max_tokens> <gpus> [split]
#
# Example:
#   bash scripts/run_pass_at_k_for_checkpoint.sh \
#     results/run/checkpoints/run_step200 \
#     gsm8k \
#     results/run/evals/gsm8k_pass128/step200 \
#     128 512 "0,1,2,3,4,5,6,7" test

set -e

CHECKPOINT=$1
DATASET=$2
OUTPUT_DIR=$3
N_COMPLETIONS=${4:-128}
MAX_TOKENS=${5:-1024}
GPUS=${6:-"0,1,2,3,4,5,6,7"}
SPLIT=${7:-test}

if [ -z "$CHECKPOINT" ] || [ -z "$DATASET" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <checkpoint> <dataset> <output_dir> [n_completions] [max_tokens] [gpus] [split]"
  exit 1
fi

# Check if already done
if [ -f "$OUTPUT_DIR/merged_summary.json" ]; then
  echo "[SKIP] Already done: $OUTPUT_DIR"
  exit 0
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "Pass@k evaluation"
echo "  Checkpoint: $CHECKPOINT"
echo "  Dataset: $DATASET (split=$SPLIT)"
echo "  N completions: $N_COMPLETIONS"
echo "  Max tokens: $MAX_TOKENS"
echo "  GPUs: $GPUS ($NUM_GPUS total)"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

PIDS=()
for i in "${!GPU_ARRAY[@]}"; do
  GPU=${GPU_ARRAY[$i]}
  echo "Launching shard $i on GPU $GPU..."
  CUDA_VISIBLE_DEVICES=$GPU /efs/rlvr-experiments/.venv/bin/python -u \
    /efs/rlvr-experiments/scripts/eval_pass_at_k.py \
    "$DATASET" \
    --split "$SPLIT" \
    --model-path "$CHECKPOINT" \
    --n "$N_COMPLETIONS" \
    --batch-size 16 \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len 2048 \
    --temperature 1.0 \
    --gpus "$GPU" \
    --gpu-index "$i" \
    --num-shards "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --verifier-workers 8 \
    > "$OUTPUT_DIR/shard_${i}.log" 2>&1 &
  PIDS+=($!)
  echo "  Started PID ${PIDS[$i]}"
done

echo ""
echo "All shards launched. Monitoring..."

# Wait for all processes
FAILED=0
for i in "${!PIDS[@]}"; do
  PID=${PIDS[$i]}
  if wait $PID; then
    echo "Shard $i (PID $PID) completed successfully"
  else
    echo "ERROR: Shard $i (PID $PID) failed"
    FAILED=$((FAILED + 1))
  fi
done

if [ $FAILED -gt 0 ]; then
  echo "ERROR: $FAILED shard(s) failed"
  exit 1
fi

echo ""
echo "Merging results..."

# Merge shard results with proper pass@k estimator
/efs/rlvr-experiments/.venv/bin/python3 -c "
import json, math
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

if not all_results:
    print('ERROR: No results found')
    exit(1)

overall_pass_rate = total_correct / total_completions if total_completions else 0
avg_pass_rate = sum(r['pass_rate'] for r in all_results) / len(all_results)

# Proper unbiased pass@k estimator
def pass_at_k_est(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

n_per = all_results[0]['num_completions']
pass_at_k = {}
for k in [1, 2, 4, 8, 16, 32, 64, 128]:
    if k > n_per:
        break
    estimates = [pass_at_k_est(r['num_completions'], r['num_correct'], k) for r in all_results]
    pass_at_k[f'pass@{k}'] = sum(estimates) / len(estimates)

# Write merged results
with open(output_dir / 'all_verification_results.jsonl', 'w') as f:
    for r in all_results:
        f.write(json.dumps(r) + '\n')

summary = {
    'dataset': '$DATASET',
    'checkpoint': '$CHECKPOINT',
    'num_prompts': len(all_results),
    'num_completions': total_completions,
    'num_correct': total_correct,
    'overall_pass_rate': overall_pass_rate,
    'avg_per_prompt_pass_rate': avg_pass_rate,
    'pass_at_k': pass_at_k,
}
with open(output_dir / 'merged_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Merged {len(all_results)} prompts from {len(list(output_dir.glob(\"shard_*\")))} shards')
print(f'Pass@k:')
for k, rate in pass_at_k.items():
    print(f'  {k}: {rate*100:.2f}%')
"

echo ""
echo "Done! Results in $OUTPUT_DIR/merged_summary.json"
