#!/bin/bash
# Master eval script - runs all evaluations for GSM8k step-200 checkpoints
# Usage: bash scripts/run_all_evals.sh
set -e

cd /efs/rlvr-experiments

# All runs that need evaluation at step 200
ALL_RUNS=(
  # LR sweep (no SFT)
  "qwen3-1.7B-gsm8k-dapo-lr1e5_20260129-222310"
  "qwen3-1.7B-gsm8k-dapo-lr5e6_20260129-213056"
  "qwen3-1.7B-gsm8k-grpo-lr1e5_20260130-012436"
  "qwen3-1.7B-gsm8k-grpo-lr5e6_20260129-213606"
  # LR sweep (with SFT)
  "qwen3-1.7B-gsm8k-dapo-sft-lr1e5_20260130-005912"
  "qwen3-1.7B-gsm8k-dapo-sft-lr5e6_20260130-005852"
  "qwen3-1.7B-gsm8k-grpo-sft-lr1e5_20260130-022119"
  "qwen3-1.7B-gsm8k-grpo-sft-lr5e6_20260130-002249"
  # Staleness ablation
  "qwen3-1.7B-gsm8k-dapo-lr5e6-stale1_20260131-005831"
  "qwen3-1.7B-gsm8k-dapo-lr5e6-stale2_20260131-052241"
  "qwen3-1.7B-gsm8k-dapo-lr5e6-stale4_20260131-070356"
  "qwen3-1.7B-gsm8k-grpo-lr5e6-stale1_20260131-010105"
  "qwen3-1.7B-gsm8k-grpo-lr5e6-stale2_20260131-051827"
  "qwen3-1.7B-gsm8k-grpo-lr5e6-stale4_20260131-070653"
)

# Function to run a pass@1 eval on a single GPU
run_pass1_eval() {
  local run=$1
  local benchmark=$2
  local gpu=$3
  local ckpt="results/$run/checkpoints/${run}_step200"
  local outdir="results/$run/evals/$benchmark/step200"
  local logfile="/tmp/eval_${benchmark}_${run}.log"

  # Check if already done
  if [ -f "$outdir/summary.json" ]; then
    echo "[SKIP] $run $benchmark - already done"
    return 0
  fi

  mkdir -p "$outdir"
  echo "[LAUNCH] $run $benchmark on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/eval_checkpoint.py \
    "$ckpt" "$outdir" --benchmark $benchmark \
    > "$logfile" 2>&1
  echo "[DONE] $run $benchmark"
}

# Function to run pass@k eval using all specified GPUs
run_pass_at_k_eval() {
  local run=$1
  local dataset=$2
  local n=$3         # number of completions
  local gpus=$4      # comma-separated GPU list
  local max_tokens=$5
  local split=${6:-test}
  local ckpt="results/$run/checkpoints/${run}_step200"
  local outdir="results/$run/evals/${dataset}_pass${n}/step200"
  local logfile="/tmp/eval_pass${n}_${dataset}_${run}.log"

  # Check if already done
  if [ -f "$outdir/merged_summary.json" ]; then
    echo "[SKIP] $run ${dataset}_pass${n} - already done"
    return 0
  fi

  mkdir -p "$outdir"
  echo "[LAUNCH] $run ${dataset} pass@${n} on GPUs $gpus"

  # Use launch_pass_at_k.sh but with custom model path
  IFS=',' read -ra GPU_ARRAY <<< "$gpus"
  NUM_GPUS=${#GPU_ARRAY[@]}

  PIDS=()
  for i in "${!GPU_ARRAY[@]}"; do
    GPU=${GPU_ARRAY[$i]}
    CUDA_VISIBLE_DEVICES=$GPU python -u scripts/eval_pass_at_k.py \
      "$dataset" \
      --split "$split" \
      --model-path "$ckpt" \
      --n "$n" \
      --batch-size 16 \
      --max-tokens "$max_tokens" \
      --max-model-len 2048 \
      --temperature 1.0 \
      --gpus "$GPU" \
      --gpu-index "$i" \
      --num-shards "$NUM_GPUS" \
      --output-dir "$outdir" \
      --verifier-workers 8 \
      > "$outdir/shard_${i}.log" 2>&1 &
    PIDS+=($!)
  done

  # Wait for all shards
  FAILED=0
  for i in "${!PIDS[@]}"; do
    if ! wait ${PIDS[$i]}; then
      echo "[ERROR] Shard $i failed for $run ${dataset} pass@${n}"
      FAILED=$((FAILED + 1))
    fi
  done

  if [ $FAILED -gt 0 ]; then
    echo "[ERROR] $FAILED shard(s) failed for $run ${dataset} pass@${n}"
    return 1
  fi

  # Merge results
  python3 -c "
import json
from pathlib import Path
import math

output_dir = Path('$outdir')
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

overall_pass_rate = total_correct / total_completions if total_completions else 0
avg_pass_rate = sum(r['pass_rate'] for r in all_results) / len(all_results) if all_results else 0

# Compute proper pass@k using the unbiased estimator
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

n_per = all_results[0]['num_completions'] if all_results else 0
pass_at_k_results = {}
for k in [1, 2, 4, 8, 16, 32, 64, 128]:
    if k > n_per:
        break
    estimates = [pass_at_k(r['num_completions'], r['num_correct'], k) for r in all_results]
    pass_at_k_results[f'pass@{k}'] = sum(estimates) / len(estimates) if estimates else 0

merged_file = output_dir / 'all_verification_results.jsonl'
with open(merged_file, 'w') as f:
    for r in all_results:
        f.write(json.dumps(r) + '\n')

summary = {
    'dataset': '$dataset',
    'num_prompts': len(all_results),
    'num_completions': total_completions,
    'num_correct': total_correct,
    'overall_pass_rate': overall_pass_rate,
    'avg_per_prompt_pass_rate': avg_pass_rate,
    'pass_at_k': pass_at_k_results,
}
with open(output_dir / 'merged_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Merged {len(all_results)} prompts')
for k, rate in pass_at_k_results.items():
    print(f'  {k}: {rate*100:.2f}%')
"

  echo "[DONE] $run ${dataset} pass@${n}"
}

echo "============================================================"
echo "Master eval script starting"
echo "============================================================"

# Phase 1: Pass@1 evals (GSM8k, MATH, MMLU) - 1 GPU each
echo ""
echo "=== Phase 1: Pass@1 evals ==="

# Run MMLU evals - these are needed and haven't been done
for run in "${ALL_RUNS[@]}"; do
  # MMLU needs to be done for all
  run_pass1_eval "$run" "mmlu" "$(( RANDOM % 8 ))" &
done
wait
echo "MMLU evals done"

# Phase 2: Pass@128 for GSM8k (use all 8 GPUs per run)
echo ""
echo "=== Phase 2: GSM8k pass@128 ==="
for run in "${ALL_RUNS[@]}"; do
  run_pass_at_k_eval "$run" "gsm8k" 128 "0,1,2,3,4,5,6,7" 512 "test"
done

# Phase 3: Pass@128 for MATH (use all 8 GPUs per run)
echo ""
echo "=== Phase 3: MATH pass@128 ==="
for run in "${ALL_RUNS[@]}"; do
  run_pass_at_k_eval "$run" "math" 128 "0,1,2,3,4,5,6,7" 1024 "test"
done

# Phase 4: AIME pass@32
echo ""
echo "=== Phase 4: AIME pass@32 ==="
for run in "${ALL_RUNS[@]}"; do
  run_pass_at_k_eval "$run" "aime" 32 "0,1,2,3,4,5,6,7" 1024 "train"
done

# Phase 5: BeyondAIME pass@32
echo ""
echo "=== Phase 5: BeyondAIME pass@32 ==="
for run in "${ALL_RUNS[@]}"; do
  run_pass_at_k_eval "$run" "beyondaime" 32 "0,1,2,3,4,5,6,7" 1024 "test"
done

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "============================================================"
