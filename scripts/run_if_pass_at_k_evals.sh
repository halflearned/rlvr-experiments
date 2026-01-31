#!/bin/bash
# Run pass@128 evaluations for all IF models on all benchmarks
# Each eval uses all 8 GPUs in parallel via sharding
set -e

PYTHON=/efs/rlvr-experiments/.venv/bin/python
SCRIPT=/efs/rlvr-experiments/scripts/launch_pass_at_k.sh
GPUS="0,1,2,3,4,5,6,7"

# Define models and their checkpoint paths
declare -A MODELS
MODELS[annotations-adhoc-20260125-083856]="/efs/rlvr-experiments/results/annotations-adhoc-20260125-083856/checkpoints/config_rewritten_20260125-084721_step200"
MODELS[qwen3-1.7B-if-dapo_20260130-101403]="/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_20260130-101403/checkpoints/qwen3-1.7B-if-dapo_20260130-101403_step200"
MODELS[qwen3-1.7B-if-dapo_sft_20260130-073008]="/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_sft_20260130-073008/checkpoints/qwen3-1.7B-if-dapo_sft_20260130-073008_step200"
MODELS[qwen3-1.7B-if-grpo_sft_20260130-223405]="/efs/rlvr-experiments/results/qwen3-1.7B-if-grpo_sft_20260130-223405/checkpoints/qwen3-1.7B-if-grpo_sft_20260130-223405_step200"

# Define benchmarks, their dataset names for pass@k, splits, and max model lengths
# Format: dataset_name:split:max_model_len:max_tokens
declare -A BENCHMARKS
BENCHMARKS[gsm8k]="gsm8k:test:2048:1024"
BENCHMARKS[math]="math:test:2048:1024"
BENCHMARKS[ifeval]="ifeval:train:4096:2048"
BENCHMARKS[ifbench]="ifbench:train:4096:2048"

for RUN_NAME in "${!MODELS[@]}"; do
    CKPT="${MODELS[$RUN_NAME]}"
    echo "================================================================"
    echo "Model: $RUN_NAME"
    echo "Checkpoint: $CKPT"
    echo "================================================================"

    for BENCH_KEY in "${!BENCHMARKS[@]}"; do
        IFS=':' read -r DATASET SPLIT MAX_MODEL_LEN MAX_TOKENS <<< "${BENCHMARKS[$BENCH_KEY]}"
        OUTPUT_DIR="/efs/rlvr-experiments/results/$RUN_NAME/evals/$BENCH_KEY/pass-at-k"

        # Skip if already done
        if [ -f "$OUTPUT_DIR/merged_summary.json" ]; then
            echo "  SKIP $BENCH_KEY pass@128 (already done)"
            continue
        fi

        echo ""
        echo "  Running $BENCH_KEY pass@128..."
        echo "  Output: $OUTPUT_DIR"
        echo "  Split: $SPLIT, Max model len: $MAX_MODEL_LEN, Max tokens: $MAX_TOKENS"

        mkdir -p "$OUTPUT_DIR"

        # Launch sharded evaluation
        PIDS=()
        IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
        NUM_GPUS=${#GPU_ARRAY[@]}

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

        # Wait for all shards
        FAILED=0
        for i in "${!PIDS[@]}"; do
            PID=${PIDS[$i]}
            if wait $PID; then
                echo "    Shard $i done"
            else
                echo "    ERROR: Shard $i failed"
                FAILED=$((FAILED + 1))
            fi
        done

        if [ $FAILED -gt 0 ]; then
            echo "  ERROR: $FAILED shard(s) failed for $BENCH_KEY"
            continue
        fi

        # Merge results
        $PYTHON -c "
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

overall_pass_rate = total_correct / total_completions if total_completions else 0
avg_pass_rate = sum(r['pass_rate'] for r in all_results) / len(all_results) if all_results else 0

n = all_results[0]['num_completions'] if all_results else 0
pass_at_k = {}
for k in [1, 2, 4, 8, 16, 32, 64, 128]:
    if k > n:
        break
    passed = sum(1 for r in all_results if any(s > 0 for s in r['scores'][:k]))
    pass_at_k[f'pass@{k}'] = passed / len(all_results) if all_results else 0

merged_file = output_dir / 'all_verification_results.jsonl'
with open(merged_file, 'w') as f:
    for r in all_results:
        f.write(json.dumps(r) + '\n')

summary = {
    'dataset': '$DATASET',
    'model': '$RUN_NAME',
    'checkpoint': '$CKPT',
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

print(f'  Merged {len(all_results)} prompts')
for k, rate in pass_at_k.items():
    print(f'    {k}: {rate*100:.2f}%')
"
        echo "  $BENCH_KEY pass@128 done"
    done
done

echo ""
echo "================================================================"
echo "All pass@128 evaluations complete!"
echo "================================================================"
