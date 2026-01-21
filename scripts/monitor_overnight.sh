#!/bin/bash
# Overnight monitoring script for v5 and v6 MATH training runs
# Watches for collapse (stuck steps) and logs progress

cd /efs/rlvr-experiments

V5_LOG="/tmp/v5_training.log"
V6_LOG="/tmp/v6_training.log"
V5_NODE="localhost"
V6_NODE="172.31.17.116"

V5_LAST_STEP=-1
V6_LAST_STEP=-1
V5_STUCK_COUNT=0
V6_STUCK_COUNT=0
STUCK_THRESHOLD=10  # 10 checks * 60s = 10 minutes stuck = collapse

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

get_current_step() {
    local log_file="$1"
    local node="$2"

    if [ "$node" = "localhost" ]; then
        grep -E '^\[epoch' "$log_file" 2>/dev/null | tail -1 | grep -oP 'step=\K\d+' || echo "-1"
    else
        ssh ubuntu@"$node" "grep -E '^\[epoch' $log_file 2>/dev/null | tail -1 | grep -oP 'step=\K\d+'" || echo "-1"
    fi
}

get_last_reward() {
    local log_file="$1"
    local node="$2"

    if [ "$node" = "localhost" ]; then
        grep -E '^\[epoch' "$log_file" 2>/dev/null | tail -1 | grep -oP 'reward=\K[0-9.]+' || echo "?"
    else
        ssh ubuntu@"$node" "grep -E '^\[epoch' $log_file 2>/dev/null | tail -1 | grep -oP 'reward=\K[0-9.]+'" || echo "?"
    fi
}

check_if_alive() {
    local node="$1"
    local search_pattern="$2"

    if [ "$node" = "localhost" ]; then
        pgrep -f "$search_pattern" > /dev/null && echo "running" || echo "stopped"
    else
        ssh ubuntu@"$node" "pgrep -f '$search_pattern'" > /dev/null 2>&1 && echo "running" || echo "stopped"
    fi
}

log "=== Starting overnight monitoring for v5 and v6 ==="
log "v5 config: lr=2e-5, beta=0.03 (aggressive)"
log "v6 config: lr=1e-5, beta=0.04 (conservative)"
log "Target: 200 steps, checkpoint every 20 steps"
log "Success: 2%+ improvement on minerva_math (base: ~27%)"
log "Collapse detection: $STUCK_THRESHOLD checks (${STUCK_THRESHOLD} minutes) at same step"
log "================================================================"

while true; do
    # Check v5
    v5_step=$(get_current_step "$V5_LOG" "$V5_NODE")
    v5_reward=$(get_last_reward "$V5_LOG" "$V5_NODE")
    v5_status=$(check_if_alive "$V5_NODE" "train_grpo.*v5")

    if [ "$v5_step" = "$V5_LAST_STEP" ] && [ "$v5_status" = "running" ]; then
        V5_STUCK_COUNT=$((V5_STUCK_COUNT + 1))
    else
        V5_STUCK_COUNT=0
    fi
    V5_LAST_STEP="$v5_step"

    # Check v6
    v6_step=$(get_current_step "$V6_LOG" "$V6_NODE")
    v6_reward=$(get_last_reward "$V6_LOG" "$V6_NODE")
    v6_status=$(check_if_alive "$V6_NODE" "train_grpo.*v6")

    if [ "$v6_step" = "$V6_LAST_STEP" ] && [ "$v6_status" = "running" ]; then
        V6_STUCK_COUNT=$((V6_STUCK_COUNT + 1))
    else
        V6_STUCK_COUNT=0
    fi
    V6_LAST_STEP="$v6_step"

    # Log status
    log "v5: step=$v5_step reward=$v5_reward status=$v5_status stuck=$V5_STUCK_COUNT"
    log "v6: step=$v6_step reward=$v6_reward status=$v6_status stuck=$V6_STUCK_COUNT"

    # Check for collapse (stuck for too long)
    if [ "$V5_STUCK_COUNT" -ge "$STUCK_THRESHOLD" ]; then
        log "WARNING: v5 appears collapsed/stuck at step $v5_step for $V5_STUCK_COUNT checks"
        log "Consider investigating or killing the run"
    fi

    if [ "$V6_STUCK_COUNT" -ge "$STUCK_THRESHOLD" ]; then
        log "WARNING: v6 appears collapsed/stuck at step $v6_step for $V6_STUCK_COUNT checks"
        log "Consider investigating or killing the run"
    fi

    # Check for completion
    if [ "$v5_step" -ge 200 ] && [ "$v6_step" -ge 200 ]; then
        log "=== BOTH RUNS COMPLETED 200 STEPS ==="
        break
    fi

    # Check for both stopped
    if [ "$v5_status" = "stopped" ] && [ "$v6_status" = "stopped" ]; then
        log "=== BOTH RUNS STOPPED ==="
        break
    fi

    echo "---"
    sleep 60
done

log "=== Monitoring complete ==="

# Final summary
log "Final v5: step=$V5_LAST_STEP"
log "Final v6: step=$V6_LAST_STEP"

# List checkpoints
log "v5 checkpoints:"
ls -la /efs/rlvr-experiments/checkpoints/ | grep "math_only_minerva_v5" || echo "  (none)"
log "v6 checkpoints:"
ls -la /efs/rlvr-experiments/checkpoints/ | grep "math_only_minerva_v6" || echo "  (none)"

# List eval results
log "v5 eval results:"
ls -la /efs/rlvr-experiments/experiments/math_only_minerva_v5/evals/*.json 2>/dev/null || echo "  (none)"
log "v6 eval results:"
ls -la /efs/rlvr-experiments/experiments/math_only_minerva_v6/evals/*.json 2>/dev/null || echo "  (none)"
