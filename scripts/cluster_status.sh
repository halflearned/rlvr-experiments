#!/bin/bash
# Cluster status checker - shows what's running on all nodes
# Usage: ./scripts/cluster_status.sh [--update]

set -e

STATUS_FILE="/efs/rlvr-experiments/CLUSTER_STATUS.md"
NODES=("localhost" "172.31.17.116" "172.31.24.124")
NODE_NAMES=("Primary" "Secondary" "Tertiary")

get_jobs() {
    local node=$1

    if [ "$node" = "localhost" ]; then
        # Get training jobs with their configs
        ps aux | grep -E "train_grpo\.py|eval_checkpoint\.py|launch_pass_at_k|eval_pass_at_k" | grep -v grep | while read line; do
            pid=$(echo "$line" | awk '{print $2}')
            start_time=$(echo "$line" | awk '{print $9}')

            # Extract config file from command
            if echo "$line" | grep -q "train_grpo.py"; then
                config=$(echo "$line" | grep -oP 'train_grpo\.py\s+\K[^\s]+' | xargs basename 2>/dev/null || echo "unknown")
                echo "TRAIN|$config|$start_time|$pid"
            elif echo "$line" | grep -q "eval_checkpoint"; then
                checkpoint=$(echo "$line" | grep -oP 'eval_checkpoint\.py\s+\K[^\s]+' | xargs basename 2>/dev/null || echo "unknown")
                benchmark=$(echo "$line" | grep -oP '\-\-benchmark\s+\K\w+' || echo "?")
                echo "EVAL|$checkpoint ($benchmark)|$start_time|$pid"
            elif echo "$line" | grep -q "pass_at_k"; then
                dataset=$(echo "$line" | grep -oP 'launch_pass_at_k\.sh\s+\K\w+' || echo "$line" | grep -oP 'eval_pass_at_k\.py\s+\K\w+' || echo "unknown")
                echo "PASSK|$dataset|$start_time|$pid"
            fi
        done
    else
        ssh -o ConnectTimeout=5 ubuntu@$node "ps aux | grep -E 'train_grpo\.py|eval_checkpoint\.py|launch_pass_at_k|eval_pass_at_k' | grep -v grep" 2>/dev/null | while read line; do
            pid=$(echo "$line" | awk '{print $2}')
            start_time=$(echo "$line" | awk '{print $9}')

            if echo "$line" | grep -q "train_grpo.py"; then
                config=$(echo "$line" | grep -oP 'train_grpo\.py\s+\K[^\s]+' | xargs basename 2>/dev/null || echo "unknown")
                echo "TRAIN|$config|$start_time|$pid"
            elif echo "$line" | grep -q "eval_checkpoint"; then
                checkpoint=$(echo "$line" | grep -oP 'eval_checkpoint\.py\s+\K[^\s]+' | xargs basename 2>/dev/null || echo "unknown")
                benchmark=$(echo "$line" | grep -oP '\-\-benchmark\s+\K\w+' || echo "?")
                echo "EVAL|$checkpoint ($benchmark)|$start_time|$pid"
            elif echo "$line" | grep -q "pass_at_k"; then
                dataset=$(echo "$line" | grep -oP 'launch_pass_at_k\.sh\s+\K\w+' || echo "$line" | grep -oP 'eval_pass_at_k\.py\s+\K\w+' || echo "unknown")
                echo "PASSK|$dataset|$start_time|$pid"
            fi
        done
    fi
}

get_gpu_summary() {
    local gpu_output=$1
    local free=""
    local busy=""
    local zombie=""

    echo "$gpu_output" | while IFS=, read -r idx mem util; do
        idx=$(echo "$idx" | tr -d ' ')
        mem=$(echo "$mem" | tr -d ' ')
        util=$(echo "$util" | tr -d ' %')

        mem_val=$(echo "$mem" | grep -oP '\d+' | head -1)

        if [ "$mem_val" -lt 100 ]; then
            echo "FREE:$idx"
        elif [ "$util" -lt 5 ]; then
            echo "ZOMBIE:$idx"
        else
            echo "BUSY:$idx"
        fi
    done
}

check_node() {
    local node=$1
    local name=$2

    if [ "$node" = "localhost" ]; then
        gpu_output=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "ERROR")
    else
        gpu_output=$(ssh -o ConnectTimeout=5 ubuntu@$node "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader" 2>/dev/null || echo "UNREACHABLE")
    fi

    # Parse GPU status
    free_gpus=""
    busy_gpus=""
    zombie_gpus=""

    while IFS=, read -r idx mem util; do
        idx=$(echo "$idx" | tr -d ' ')
        mem=$(echo "$mem" | tr -d ' ')
        util=$(echo "$util" | tr -d ' %')
        mem_val=$(echo "$mem" | grep -oP '\d+' | head -1)

        if [ -z "$mem_val" ]; then continue; fi

        if [ "$mem_val" -lt 100 ]; then
            free_gpus="$free_gpus$idx,"
        elif [ "$util" -lt 5 ]; then
            zombie_gpus="$zombie_gpus$idx,"
        else
            busy_gpus="$busy_gpus$idx,"
        fi
    done <<< "$gpu_output"

    # Remove trailing commas
    free_gpus=$(echo "$free_gpus" | sed 's/,$//')
    busy_gpus=$(echo "$busy_gpus" | sed 's/,$//')
    zombie_gpus=$(echo "$zombie_gpus" | sed 's/,$//')

    # Get jobs
    jobs=$(get_jobs "$node")

    echo "### $name ($node)"
    echo ""
    echo "| Status | GPUs |"
    echo "|--------|------|"
    [ -n "$free_gpus" ] && echo "| Free | $free_gpus |"
    [ -n "$busy_gpus" ] && echo "| Busy | $busy_gpus |"
    [ -n "$zombie_gpus" ] && echo "| Zombie (0% util) | $zombie_gpus |"
    echo ""

    if [ -n "$jobs" ]; then
        echo "**Active Jobs:**"
        echo ""
        echo "| Type | Config/Target | Started | PID |"
        echo "|------|---------------|---------|-----|"
        echo "$jobs" | while IFS='|' read -r type config started pid; do
            [ -n "$type" ] && echo "| $type | $config | $started | $pid |"
        done
        echo ""
    else
        echo "_No jobs detected_"
        echo ""
    fi
}

generate_status() {
    echo "# Cluster Status"
    echo ""
    echo "**Updated:** $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""
    echo "---"
    echo ""

    for i in "${!NODES[@]}"; do
        check_node "${NODES[$i]}" "${NODE_NAMES[$i]}"
        echo "---"
        echo ""
    done

    # Quick summary
    echo "## Quick Summary"
    echo ""
    total_free=0
    total_busy=0
    for node in "${NODES[@]}"; do
        if [ "$node" = "localhost" ]; then
            gpu_output=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)
        else
            gpu_output=$(ssh -o ConnectTimeout=5 ubuntu@$node "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader" 2>/dev/null)
        fi

        while IFS=, read -r idx mem util; do
            mem_val=$(echo "$mem" | grep -oP '\d+' | head -1)
            if [ -n "$mem_val" ] && [ "$mem_val" -lt 100 ]; then
                ((total_free++)) || true
            elif [ -n "$mem_val" ]; then
                ((total_busy++)) || true
            fi
        done <<< "$gpu_output"
    done

    echo "- **Free GPUs:** $total_free / 24"
    echo "- **Busy GPUs:** $total_busy / 24"
    echo ""

    # Recent results
    echo "## Recent Results (last 6h)"
    echo "\`\`\`"
    find /efs/rlvr-experiments/results -maxdepth 1 -type d -mmin -360 -printf "%T+ %p\n" 2>/dev/null | sort -r | head -5 | awk '{print $1, $2}' | sed 's|/efs/rlvr-experiments/results/||' || echo "None"
    echo "\`\`\`"
}

if [ "$1" = "--update" ]; then
    generate_status > "$STATUS_FILE"
    echo "Updated $STATUS_FILE"
else
    generate_status
fi
