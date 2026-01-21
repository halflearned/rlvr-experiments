#!/bin/bash
# Wait for secondary node to become free, then run evaluations
# Run with: nohup ./scripts/wait_and_eval.sh > /tmp/wait_and_eval.log 2>&1 &

SECONDARY_NODE="172.31.17.116"
CHECK_INTERVAL=60  # seconds
LOG_FILE="/tmp/wait_and_eval.log"

echo "$(date): Starting monitor for secondary node $SECONDARY_NODE"
echo "Checking every $CHECK_INTERVAL seconds..."

while true; do
    # Check GPU utilization on secondary node
    GPU_UTIL=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$SECONDARY_NODE \
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | head -1)

    if [ -z "$GPU_UTIL" ]; then
        echo "$(date): Cannot reach secondary node, retrying..."
        sleep $CHECK_INTERVAL
        continue
    fi

    # If memory used is less than 1GB, node is free
    if [ "$GPU_UTIL" -lt 1000 ]; then
        echo "$(date): Secondary node is FREE (GPU mem: ${GPU_UTIL}MB)"
        echo "Starting evaluations on secondary node..."

        # Copy eval script to secondary node and run
        ssh ubuntu@$SECONDARY_NODE "cd /efs/rlvr-experiments && source .venv/bin/activate && nohup ./scripts/eval_gsm8k_math_boxed_checkpoints.sh > /tmp/eval_gsm8k_math_boxed.log 2>&1 &"

        echo "$(date): Evaluations started on secondary node"
        echo "Monitor with: ssh ubuntu@$SECONDARY_NODE 'tail -f /tmp/eval_gsm8k_math_boxed.log'"
        break
    else
        echo "$(date): Secondary node busy (GPU mem: ${GPU_UTIL}MB), waiting..."
    fi

    sleep $CHECK_INTERVAL
done

echo "$(date): Monitor script complete"
