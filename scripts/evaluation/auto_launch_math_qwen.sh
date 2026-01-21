#!/bin/bash
# Auto-launch MATH Qwen workers as GPUs become available
# Runs continuously until all jobs are complete

JOBS_FILE="/efs/rlvr-experiments/scripts/evaluation/math_qwen_jobs.json"
EVAL_SCRIPT="/efs/rlvr-experiments/scripts/evaluation/run_lm_eval.sh"
OUTPUT_DIR="/efs/rlvr-experiments/eval_results_batch"

TOTAL_JOBS=$(python3 -c "import json; print(len(json.load(open('$JOBS_FILE'))))")
echo "Total jobs: $TOTAL_JOBS"

while true; do
    # Count completed jobs
    COMPLETED=$(ls $OUTPUT_DIR/*_math_qwen_4shot.json 2>/dev/null | wc -l)
    
    if [ "$COMPLETED" -ge "$TOTAL_JOBS" ]; then
        echo "All $TOTAL_JOBS jobs completed!"
        break
    fi
    
    echo "$(date): $COMPLETED/$TOTAL_JOBS completed"
    
    # Check for free GPUs (less than 1GB used = free)
    for GPU in 0 1 2 3 4 5 6 7; do
        MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
        
        if [ "$MEM_USED" -lt 1000 ]; then
            echo "GPU $GPU is free (${MEM_USED}MB used)"
            
            # Find next incomplete job
            for JOB_IDX in $(seq 0 $((TOTAL_JOBS-1))); do
                JOB_INFO=$(python3 -c "import json; j=json.load(open('$JOBS_FILE'))[$JOB_IDX]; print(j['checkpoint_path'] + '|' + j['output_name'])")
                CKPT_PATH="${JOB_INFO%|*}"
                OUTPUT_NAME="${JOB_INFO#*|}"
                
                OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_math_qwen_4shot.json"
                if [ ! -f "$OUTPUT_FILE" ]; then
                    # Check if already running
                    if ! pgrep -f "$OUTPUT_NAME.*math_qwen" > /dev/null; then
                        echo "Starting job $JOB_IDX ($OUTPUT_NAME) on GPU $GPU"
                        nohup $EVAL_SCRIPT "$CKPT_PATH" "$OUTPUT_NAME" "math_qwen" "4" "$GPU" \
                            > /efs/rlvr-experiments/eval_logs/math_qwen_${OUTPUT_NAME}_gpu${GPU}.log 2>&1 &
                        break
                    fi
                fi
            done
        fi
    done
    
    sleep 60
done
