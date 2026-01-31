#!/bin/bash
# Run remaining pass@128 evaluations
set -e

PYTHON=/efs/rlvr-experiments/.venv/bin/python
GPUS="0,1,2,3,4,5,6,7"
export PYTHONPATH=/efs/rlvr-experiments/src

# Models that need gsm8k/math pass@128
declare -A MODELS
MODELS[qwen3-1.7B-if-dapo_20260130-101403]="/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_20260130-101403/checkpoints/qwen3-1.7B-if-dapo_20260130-101403_step200"
MODELS[qwen3-1.7B-if-dapo_sft_20260130-073008]="/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_sft_20260130-073008/checkpoints/qwen3-1.7B-if-dapo_sft_20260130-073008_step200"
MODELS[qwen3-1.7B-if-grpo_sft_20260130-223405]="/efs/rlvr-experiments/results/qwen3-1.7B-if-grpo_sft_20260130-223405/checkpoints/qwen3-1.7B-if-grpo_sft_20260130-223405_step200"

# ALL models need ifeval/ifbench pass@128 (including GRPO)
declare -A ALL_MODELS
ALL_MODELS[annotations-adhoc-20260125-083856]="/efs/rlvr-experiments/results/annotations-adhoc-20260125-083856/checkpoints/config_rewritten_20260125-084721_step200"
ALL_MODELS[qwen3-1.7B-if-dapo_20260130-101403]="/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_20260130-101403/checkpoints/qwen3-1.7B-if-dapo_20260130-101403_step200"
ALL_MODELS[qwen3-1.7B-if-dapo_sft_20260130-073008]="/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_sft_20260130-073008/checkpoints/qwen3-1.7B-if-dapo_sft_20260130-073008_step200"
ALL_MODELS[qwen3-1.7B-if-grpo_sft_20260130-223405]="/efs/rlvr-experiments/results/qwen3-1.7B-if-grpo_sft_20260130-223405/checkpoints/qwen3-1.7B-if-grpo_sft_20260130-223405_step200"

run_pass_at_k_sharded() {
    local CKPT="$1"
    local DATASET="$2"
    local SPLIT="$3"
    local MAX_MODEL_LEN="$4"
    local MAX_TOKENS="$5"
    local OUTPUT_DIR="$6"

    if [ -f "$OUTPUT_DIR/merged_summary.json" ]; then
        echo "  SKIP (already done)"
        return 0
    fi

    mkdir -p "$OUTPUT_DIR"
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    local NUM_GPUS=${#GPU_ARRAY[@]}
    local PIDS=()

    for i in "${!GPU_ARRAY[@]}"; do
        GPU=${GPU_ARRAY[$i]}
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
            /efs/rlvr-experiments/scripts/eval_pass_at_k.py \
            "$DATASET" \
            --split "$SPLIT" \
            --model-path "$CKPT" \
            --n 128 \
            --batch-size 16 \
            --max-tokens "$MAX_TOKENS" \
            --max-model-len "$MAX_MODEL_LEN" \
            --temperature 1.0 \
            --gpus "$GPU" \
            --gpu-index "$i" \
            --num-shards "$NUM_GPUS" \
            --output-dir "$OUTPUT_DIR" \
            --verifier-workers 8 \
            > "$OUTPUT_DIR/shard_${i}.log" 2>&1 &
        PIDS+=($!)
    done

    local FAILED=0
    for i in "${!PIDS[@]}"; do
        if ! wait ${PIDS[$i]}; then
            echo "    Shard $i FAILED"
            FAILED=$((FAILED + 1))
        else
            echo "    Shard $i done"
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo "  ERROR: $FAILED shards failed"
        return 1
    fi

    # Merge
    $PYTHON -c "
import json
from pathlib import Path
output_dir = Path('$OUTPUT_DIR')
all_results = []
total_correct = total_completions = 0
for shard_dir in sorted(output_dir.glob('shard_*')):
    results_file = shard_dir / 'verification_results.jsonl'
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    all_results.append(r)
                    total_correct += r['num_correct']
                    total_completions += r['num_completions']

n = all_results[0]['num_completions'] if all_results else 0
pass_at_k = {}
for k in [1, 2, 4, 8, 16, 32, 64, 128]:
    if k > n: break
    passed = sum(1 for r in all_results if any(s > 0 for s in r['scores'][:k]))
    pass_at_k[f'pass@{k}'] = passed / len(all_results) if all_results else 0

with open(output_dir / 'all_verification_results.jsonl', 'w') as f:
    for r in all_results: f.write(json.dumps(r) + '\n')

summary = {'num_prompts': len(all_results), 'pass_at_k': pass_at_k,
           'overall_pass_rate': total_correct / total_completions if total_completions else 0}
with open(output_dir / 'merged_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

for k, rate in pass_at_k.items(): print(f'    {k}: {rate*100:.2f}%')
"
    echo "  DONE"
}

# === Phase 1: gsm8k/math pass@128 for 3 models ===
for RUN_NAME in "${!MODELS[@]}"; do
    CKPT="${MODELS[$RUN_NAME]}"
    echo "================================================================"
    echo "Phase 1: $RUN_NAME - gsm8k pass@128"
    echo "================================================================"
    run_pass_at_k_sharded "$CKPT" gsm8k test 2048 1024 \
        "/efs/rlvr-experiments/results/$RUN_NAME/evals/gsm8k/pass-at-k"

    echo "================================================================"
    echo "Phase 1: $RUN_NAME - math pass@128"
    echo "================================================================"
    run_pass_at_k_sharded "$CKPT" math test 2048 1024 \
        "/efs/rlvr-experiments/results/$RUN_NAME/evals/math/pass-at-k"
done

# === Phase 2: ifeval/ifbench pass@128 for ALL 4 models ===
# Use single-GPU eval_pass_at_k_test.py (one model per GPU)
echo "================================================================"
echo "Phase 2: IFEval/IFBench pass@128 for all 4 models"
echo "================================================================"

GPU_IDX=0
PIDS=()
for RUN_NAME in "${!ALL_MODELS[@]}"; do
    CKPT="${ALL_MODELS[$RUN_NAME]}"

    # IFEval pass@128
    OUTPUT_IFEVAL="/efs/rlvr-experiments/results/$RUN_NAME/evals/ifeval/pass-at-k"
    if [ ! -f "$OUTPUT_IFEVAL/merged_summary.json" ]; then
        echo "  $RUN_NAME ifeval pass@128 on GPU $GPU_IDX"
        $PYTHON -u /efs/rlvr-experiments/scripts/eval_pass_at_k_test.py \
            --benchmark ifeval \
            --model-path "$CKPT" \
            --output-dir "$OUTPUT_IFEVAL" \
            --n 128 --gpu $GPU_IDX \
            > "$OUTPUT_IFEVAL/eval.log" 2>&1 &
        PIDS+=($!)
        GPU_IDX=$((GPU_IDX + 1))
    fi

    # IFBench pass@128
    OUTPUT_IFBENCH="/efs/rlvr-experiments/results/$RUN_NAME/evals/ifbench_test/pass-at-k"
    if [ ! -f "$OUTPUT_IFBENCH/merged_summary.json" ]; then
        echo "  $RUN_NAME ifbench pass@128 on GPU $GPU_IDX"
        mkdir -p "$OUTPUT_IFBENCH"
        $PYTHON -u /efs/rlvr-experiments/scripts/eval_pass_at_k_test.py \
            --benchmark ifbench \
            --model-path "$CKPT" \
            --output-dir "$OUTPUT_IFBENCH" \
            --n 128 --gpu $GPU_IDX \
            > "$OUTPUT_IFBENCH/eval.log" 2>&1 &
        PIDS+=($!)
        GPU_IDX=$((GPU_IDX + 1))
    fi

    # If we've used all 8 GPUs, wait for them
    if [ $GPU_IDX -ge 8 ]; then
        echo "  Waiting for batch to finish..."
        for PID in "${PIDS[@]}"; do
            wait $PID || echo "  WARNING: PID $PID failed"
        done
        GPU_IDX=0
        PIDS=()
    fi
done

# Wait for remaining
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "  Waiting for final batch..."
    for PID in "${PIDS[@]}"; do
        wait $PID || echo "  WARNING: PID $PID failed"
    done
fi

echo ""
echo "================================================================"
echo "All remaining pass@128 evaluations complete!"
echo "================================================================"
