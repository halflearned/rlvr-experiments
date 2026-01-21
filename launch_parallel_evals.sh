#!/bin/bash
# Launch all evals in parallel across 8 GPUs

pkill -9 -f 'run_batch_evals' 2>/dev/null || true
pkill -9 -f 'lm_eval' 2>/dev/null || true
sleep 2

SCRIPT=/efs/rlvr-experiments/scripts/evaluation/run_lm_eval.sh
CKPTS=("math_only_minerva_step80" "math_only_minerva_step100")
TASKS=("gsm8k:4" "gsm8k_cot:4" "minerva_math:4" "ifeval:0" "mbpp:3")

gpu=0
for ckpt in "${CKPTS[@]}"; do
  for task_spec in "${TASKS[@]}"; do
    task="${task_spec%%:*}"
    fewshot="${task_spec##*:}"
    echo "Launching $ckpt $task on GPU $gpu"
    nohup $SCRIPT /efs/rlvr-experiments/checkpoints/$ckpt $ckpt $task $fewshot $gpu > /efs/rlvr-experiments/eval_logs/${ckpt}_${task}.log 2>&1 &
    gpu=$(( (gpu + 1) % 8 ))
  done
done

echo "Launched 10 eval jobs across 8 GPUs"
