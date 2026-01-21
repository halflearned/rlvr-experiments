#!/bin/bash
# Launch MATH Qwen workers on specified GPUs
# Usage: ./launch_math_qwen_workers.sh <start_job_idx> <num_jobs> [first_gpu]

START_IDX=${1:-0}
NUM_JOBS=${2:-8}
FIRST_GPU=${3:-0}

JOBS_FILE="/efs/rlvr-experiments/scripts/evaluation/math_qwen_jobs.json"
EVAL_SCRIPT="/efs/rlvr-experiments/scripts/evaluation/run_lm_eval.sh"

TOTAL_JOBS=$(python3 -c "import json; print(len(json.load(open('$JOBS_FILE'))))")
echo "Total jobs: $TOTAL_JOBS"
echo "Starting from job $START_IDX, running $NUM_JOBS jobs on GPUs starting at $FIRST_GPU"

for i in $(seq 0 $((NUM_JOBS-1))); do
    JOB_IDX=$((START_IDX + i))
    GPU=$((FIRST_GPU + i))
    
    if [ $JOB_IDX -ge $TOTAL_JOBS ]; then
        echo "No more jobs"
        break
    fi
    
    JOB=$(python3 -c "import json; jobs=json.load(open('$JOBS_FILE')); print(jobs[$JOB_IDX]['checkpoint_path'])")
    NAME=$(python3 -c "import json; jobs=json.load(open('$JOBS_FILE')); print(jobs[$JOB_IDX]['output_name'])")
    
    # Check if already done
    if [ -f "/efs/rlvr-experiments/eval_results_batch/${NAME}_math_qwen_4shot.json" ]; then
        echo "SKIP: Job $JOB_IDX ($NAME) already completed"
        continue
    fi
    
    nohup $EVAL_SCRIPT "$JOB" "$NAME" "math_qwen" "4" "$GPU" \
        > /efs/rlvr-experiments/eval_logs/math_qwen_job${JOB_IDX}_gpu${GPU}.log 2>&1 &
    echo "Started job $JOB_IDX ($NAME) on GPU $GPU, PID: $!"
done
