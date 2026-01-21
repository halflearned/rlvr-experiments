#!/bin/bash
# Launch 4 parallel eval workers on remote node (172.31.17.116)
# Jobs 88-174 go to remote node (second half)

cd /efs/rlvr-experiments/scripts

echo "Launching 4 workers on remote node (jobs 88-174)"

# Worker 4: GPUs 0,1, jobs 88-109
nohup ./eval_worker.sh 4 "0,1" 88 109 > /efs/rlvr-experiments/eval_logs/worker_remote_0.log 2>&1 &
echo "Worker 4 (GPUs 0,1) PID: $!"

# Worker 5: GPUs 2,3, jobs 110-131
nohup ./eval_worker.sh 5 "2,3" 110 131 > /efs/rlvr-experiments/eval_logs/worker_remote_1.log 2>&1 &
echo "Worker 5 (GPUs 2,3) PID: $!"

# Worker 6: GPUs 4,5, jobs 132-153
nohup ./eval_worker.sh 6 "4,5" 132 153 > /efs/rlvr-experiments/eval_logs/worker_remote_2.log 2>&1 &
echo "Worker 6 (GPUs 4,5) PID: $!"

# Worker 7: GPUs 6,7, jobs 154-174
nohup ./eval_worker.sh 7 "6,7" 154 174 > /efs/rlvr-experiments/eval_logs/worker_remote_3.log 2>&1 &
echo "Worker 7 (GPUs 6,7) PID: $!"

echo "All remote workers launched. Monitor with: tail -f /efs/rlvr-experiments/eval_logs/worker_remote_*.log"
