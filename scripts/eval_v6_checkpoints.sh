#!/bin/bash
# Eval script for v6 checkpoints (beta=0.04, lr=1e-5)
# Run on primary node using GPU 7 (or on secondary node if available)

set -e
cd /efs/rlvr-experiments

CHECKPOINT_DIR="/efs/rlvr-experiments/checkpoints"
RESULTS_DIR="/efs/rlvr-experiments/experiments/math_only_minerva_v6/evals"
mkdir -p "$RESULTS_DIR"

# Use GPU 7
GPU=7

export HF_ALLOW_CODE_EVAL=1

# Watch for new checkpoints and run evals
while true; do
    # Find v6 checkpoints that haven't been evaluated yet
    for ckpt in $(ls "$CHECKPOINT_DIR" 2>/dev/null | grep "math_only_minerva_v6_step" | sort -V); do
        step=$(echo "$ckpt" | grep -oP 'step\K\d+')

        # Skip if already evaluated (check for timestamped results using glob)
        if ls "$RESULTS_DIR"/minerva_math_step${step}_*.json 1>/dev/null 2>&1; then
            continue
        fi

        echo "=== Evaluating checkpoint: $ckpt (step $step) ==="
        ckpt_path="$CHECKPOINT_DIR/$ckpt"

        # Rename model.safetensors to expected name if needed (vLLM expects sharded naming)
        if [ -f "$ckpt_path/model.safetensors" ]; then
            mv "$ckpt_path/model.safetensors" "$ckpt_path/model-00001-of-00001.safetensors"
            echo "Renamed model weights in $ckpt_path"
        fi

        # Verify weights exist before running eval
        if [ ! -f "$ckpt_path/model-00001-of-00001.safetensors" ]; then
            echo "ERROR: No model weights found in $ckpt_path"
            continue
        fi

        echo "[$(date)] Running minerva_math on GPU $GPU..."
        CUDA_VISIBLE_DEVICES=$GPU uv run lm_eval --model vllm \
            --model_args pretrained=$ckpt_path,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096 \
            --tasks minerva_math --num_fewshot 4 --batch_size auto --seed 42 \
            --gen_kwargs temperature=0 \
            --output_path "$RESULTS_DIR/minerva_math_step${step}.json" 2>&1 | tee "$RESULTS_DIR/minerva_math_step${step}.log"

        # Extract and print result (find the timestamped JSON file)
        result_file=$(ls -t "$RESULTS_DIR"/minerva_math_step${step}_*.json 2>/dev/null | head -1)
        if [ -n "$result_file" ]; then
            result=$(python3 -c "
import json
d = json.load(open('$result_file'))
res = d.get('results', {})
for k, v in res.items():
    if 'exact_match,none' in v:
        print(f'minerva_math step $step: {v[\"exact_match,none\"]*100:.2f}%')
        break
" 2>/dev/null || echo "minerva_math step ${step}: parse error")
            echo "$result"
        else
            echo "ERROR: No result file found for step $step"
        fi
        echo ""
    done

    echo "[$(date)] Waiting 120s for new checkpoints..."
    sleep 120
done
