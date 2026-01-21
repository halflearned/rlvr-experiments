#!/bin/bash
# Launch minerva_math evals with --log_samples for all checkpoints
# STAGGERED START to avoid GPU contention

SCRIPT=/efs/rlvr-experiments/scripts/evaluation/run_lm_eval.sh

echo "Starting staggered minerva_math evals..."

# GPU 0: Base - start immediately
echo "GPU 0: Base"
nohup $SCRIPT /efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base base minerva_math 4 0 > /efs/rlvr-experiments/eval_logs/base_minerva_math_samples.log 2>&1 &
sleep 10

# GPU 1: Step 20
echo "GPU 1: Step 20"
nohup $SCRIPT /efs/rlvr-experiments/checkpoints/math_only_minerva_step20 math_only_minerva_step20 minerva_math 4 1 > /efs/rlvr-experiments/eval_logs/step20_minerva_math_samples.log 2>&1 &
sleep 10

# GPU 2: Step 40
echo "GPU 2: Step 40"
nohup $SCRIPT /efs/rlvr-experiments/checkpoints/math_only_minerva_step40 math_only_minerva_step40 minerva_math 4 2 > /efs/rlvr-experiments/eval_logs/step40_minerva_math_samples.log 2>&1 &
sleep 10

# GPU 3: Step 60
echo "GPU 3: Step 60"
nohup $SCRIPT /efs/rlvr-experiments/checkpoints/math_only_minerva_step60 math_only_minerva_step60 minerva_math 4 3 > /efs/rlvr-experiments/eval_logs/step60_minerva_math_samples.log 2>&1 &
sleep 10

# GPU 4: Step 80
echo "GPU 4: Step 80"
nohup $SCRIPT /efs/rlvr-experiments/checkpoints/math_only_minerva_step80 math_only_minerva_step80 minerva_math 4 4 > /efs/rlvr-experiments/eval_logs/step80_minerva_math_samples.log 2>&1 &
sleep 10

# GPU 5: Step 100
echo "GPU 5: Step 100"
nohup $SCRIPT /efs/rlvr-experiments/checkpoints/math_only_minerva_step100 math_only_minerva_step100 minerva_math 4 5 > /efs/rlvr-experiments/eval_logs/step100_minerva_math_samples.log 2>&1 &

echo "Launched 6 minerva_math evals with --log_samples on GPUs 0-5 (staggered 10s apart)"
