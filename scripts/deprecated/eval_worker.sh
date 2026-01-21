#!/bin/bash
# Eval worker - runs evals from pending_evals.json
# Usage: ./eval_worker.sh <worker_id> <gpu_pair> [start_idx] [end_idx]
# Example: ./eval_worker.sh 0 "0,1" 0 43  (run jobs 0-43 on GPUs 0,1)

set -e

WORKER_ID=$1
GPU_PAIR=$2
START_IDX=${3:-0}
END_IDX=${4:-999999}

EVAL_BIN="/efs/rlvr-experiments/.venv/bin/lm_eval"
OUTPUT_DIR="/efs/rlvr-experiments/eval_results_batch"
JOBS_FILE="/efs/rlvr-experiments/scripts/pending_evals.json"
LOG_DIR="/efs/rlvr-experiments/eval_logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Enable code evaluation for MBPP
export HF_ALLOW_CODE_EVAL=1

# Read jobs from JSON
JOBS=$(cat "$JOBS_FILE")
TOTAL=$(echo "$JOBS" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")

echo "Worker $WORKER_ID starting on GPUs $GPU_PAIR"
echo "Processing jobs $START_IDX to $END_IDX (total available: $TOTAL)"

for i in $(seq $START_IDX $END_IDX); do
    if [ $i -ge $TOTAL ]; then
        echo "No more jobs"
        break
    fi
    
    # Extract job info
    JOB=$(echo "$JOBS" | python3 -c "import json,sys; j=json.load(sys.stdin)[$i]; print(json.dumps(j))")
    CKPT_PATH=$(echo "$JOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['checkpoint_path'])")
    TASK=$(echo "$JOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['task'])")
    FEWSHOT=$(echo "$JOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['num_fewshot'])")
    OUTPUT_NAME=$(echo "$JOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_name'])")
    
    # Skip if already done
    if [ -d "$OUTPUT_DIR/$OUTPUT_NAME" ] && find "$OUTPUT_DIR/$OUTPUT_NAME" -name "results*.json" | grep -q .; then
        echo "[$i] SKIP $OUTPUT_NAME (already done)"
        continue
    fi
    
    # Fix safetensors filename if needed
    if [ -f "$CKPT_PATH/model.safetensors" ] && [ ! -f "$CKPT_PATH/model-00001-of-00001.safetensors" ]; then
        echo "[$i] Renaming model.safetensors -> model-00001-of-00001.safetensors"
        mv "$CKPT_PATH/model.safetensors" "$CKPT_PATH/model-00001-of-00001.safetensors"
    fi
    
    echo "[$i] Running $OUTPUT_NAME on GPUs $GPU_PAIR"

    # Add --confirm_run_unsafe_code for mbpp/humaneval
    EXTRA_ARGS=""
    if [ "$TASK" = "mbpp" ] || [ "$TASK" = "humaneval" ]; then
        EXTRA_ARGS="--confirm_run_unsafe_code"
    fi

    CUDA_VISIBLE_DEVICES=$GPU_PAIR $EVAL_BIN --model vllm \
        --model_args "pretrained=$CKPT_PATH,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=4096,seed=42" \
        --tasks "$TASK" \
        --num_fewshot "$FEWSHOT" \
        --batch_size auto \
        --seed 42 \
        --gen_kwargs temperature=0 \
        $EXTRA_ARGS \
        --output_path "$OUTPUT_DIR/$OUTPUT_NAME" \
        2>&1 | tee "$LOG_DIR/${OUTPUT_NAME}.log"
    
    echo "[$i] Done $OUTPUT_NAME"
done

echo "Worker $WORKER_ID finished"
