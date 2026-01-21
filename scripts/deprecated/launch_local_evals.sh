#!/bin/bash
# Launch 4 parallel eval workers on local node
# Jobs 0-87 go to local node (first half)

cd /efs/rlvr-experiments/scripts

echo "Launching 4 workers on local node (jobs 0-87)"

# Worker 0: GPUs 0,1, jobs 0-21
nohup ./eval_worker.sh 0 "0,1" 0 21 > /efs/rlvr-experiments/eval_logs/worker_local_0.log 2>&1 &
echo "Worker 0 (GPUs 0,1) PID: $!"

# Worker 1: GPUs 2,3, jobs 22-43
nohup ./eval_worker.sh 1 "2,3" 22 43 > /efs/rlvr-experiments/eval_logs/worker_local_1.log 2>&1 &
echo "Worker 1 (GPUs 2,3) PID: $!"

# Worker 2: GPUs 4,5, jobs 44-65
nohup ./eval_worker.sh 2 "4,5" 44 65 > /efs/rlvr-experiments/eval_logs/worker_local_2.log 2>&1 &
echo "Worker 2 (GPUs 4,5) PID: $!"

# Worker 3: GPUs 6,7, jobs 66-87
nohup ./eval_worker.sh 3 "6,7" 66 87 > /efs/rlvr-experiments/eval_logs/worker_local_3.log 2>&1 &
echo "Worker 3 (GPUs 6,7) PID: $!"

echo "All local workers launched. Monitor with: tail -f /efs/rlvr-experiments/eval_logs/worker_local_*.log"
