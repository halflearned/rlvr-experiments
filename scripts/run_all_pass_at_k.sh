#!/bin/bash
# Run all pass@k evaluations for GSM8k step-200 checkpoints
# This script runs sequentially (one checkpoint at a time) using all 8 GPUs
set -e

cd /efs/rlvr-experiments

ALL_RUNS=(
  "qwen3-1.7B-gsm8k-dapo-lr1e5_20260129-222310"
  "qwen3-1.7B-gsm8k-dapo-lr5e6_20260129-213056"
  "qwen3-1.7B-gsm8k-grpo-lr1e5_20260130-012436"
  "qwen3-1.7B-gsm8k-grpo-lr5e6_20260129-213606"
  "qwen3-1.7B-gsm8k-dapo-sft-lr1e5_20260130-005912"
  "qwen3-1.7B-gsm8k-dapo-sft-lr5e6_20260130-005852"
  "qwen3-1.7B-gsm8k-grpo-sft-lr1e5_20260130-022119"
  "qwen3-1.7B-gsm8k-grpo-sft-lr5e6_20260130-002249"
  "qwen3-1.7B-gsm8k-dapo-lr5e6-stale1_20260131-005831"
  "qwen3-1.7B-gsm8k-dapo-lr5e6-stale2_20260131-052241"
  "qwen3-1.7B-gsm8k-dapo-lr5e6-stale4_20260131-070356"
  "qwen3-1.7B-gsm8k-grpo-lr5e6-stale1_20260131-010105"
  "qwen3-1.7B-gsm8k-grpo-lr5e6-stale2_20260131-051827"
  "qwen3-1.7B-gsm8k-grpo-lr5e6-stale4_20260131-070653"
)

GPUS="0,1,2,3,4,5,6,7"

echo "============================================================"
echo "Starting all pass@k evaluations at $(date)"
echo "============================================================"

# Phase 1: GSM8k pass@128 (512 max tokens, test split)
echo ""
echo "=== Phase 1: GSM8k pass@128 ==="
for run in "${ALL_RUNS[@]}"; do
  ckpt="results/$run/checkpoints/${run}_step200"
  outdir="results/$run/evals/gsm8k_pass128/step200"
  if [ -f "$outdir/merged_summary.json" ]; then
    echo "[SKIP] $run - gsm8k_pass128 already done"
    continue
  fi
  echo ""
  echo "[$(date '+%H:%M:%S')] Processing $run - gsm8k pass@128"
  bash scripts/run_pass_at_k_for_checkpoint.sh "$ckpt" gsm8k "$outdir" 128 512 "$GPUS" test 2>&1 | tee -a /tmp/all_pass_at_k.log
done

# Phase 2: MATH pass@128 (1024 max tokens, test split)
echo ""
echo "=== Phase 2: MATH pass@128 ==="
for run in "${ALL_RUNS[@]}"; do
  ckpt="results/$run/checkpoints/${run}_step200"
  outdir="results/$run/evals/math_pass128/step200"
  if [ -f "$outdir/merged_summary.json" ]; then
    echo "[SKIP] $run - math_pass128 already done"
    continue
  fi
  echo ""
  echo "[$(date '+%H:%M:%S')] Processing $run - math pass@128"
  bash scripts/run_pass_at_k_for_checkpoint.sh "$ckpt" math "$outdir" 128 1024 "$GPUS" test 2>&1 | tee -a /tmp/all_pass_at_k.log
done

# Phase 3: AIME pass@32 (1024 max tokens, train split - only 90 examples)
echo ""
echo "=== Phase 3: AIME pass@32 ==="
for run in "${ALL_RUNS[@]}"; do
  ckpt="results/$run/checkpoints/${run}_step200"
  outdir="results/$run/evals/aime_pass32/step200"
  if [ -f "$outdir/merged_summary.json" ]; then
    echo "[SKIP] $run - aime_pass32 already done"
    continue
  fi
  echo ""
  echo "[$(date '+%H:%M:%S')] Processing $run - aime pass@32"
  bash scripts/run_pass_at_k_for_checkpoint.sh "$ckpt" aime "$outdir" 32 1024 "$GPUS" train 2>&1 | tee -a /tmp/all_pass_at_k.log
done

# Phase 4: BeyondAIME pass@32 (1024 max tokens, test split - 100 examples)
echo ""
echo "=== Phase 4: BeyondAIME pass@32 ==="
for run in "${ALL_RUNS[@]}"; do
  ckpt="results/$run/checkpoints/${run}_step200"
  outdir="results/$run/evals/beyondaime_pass32/step200"
  if [ -f "$outdir/merged_summary.json" ]; then
    echo "[SKIP] $run - beyondaime_pass32 already done"
    continue
  fi
  echo ""
  echo "[$(date '+%H:%M:%S')] Processing $run - beyondaime pass@32"
  bash scripts/run_pass_at_k_for_checkpoint.sh "$ckpt" beyondaime "$outdir" 32 1024 "$GPUS" test 2>&1 | tee -a /tmp/all_pass_at_k.log
done

echo ""
echo "============================================================"
echo "All pass@k evaluations complete at $(date)"
echo "============================================================"
