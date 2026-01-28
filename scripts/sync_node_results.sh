#!/bin/bash
# Sync results from a remote node to EFS
# Usage: ./scripts/sync_node_results.sh <node_ip> <run_name> [interval_seconds] [--with-checkpoints]
#
# Examples:
#   ./scripts/sync_node_results.sh 172.31.24.124 math-run 60                    # traces only
#   ./scripts/sync_node_results.sh 172.31.24.124 math-run 60 --with-checkpoints # include model weights

set -e

NODE_IP="${1:?Usage: $0 <node_ip> <run_name> [interval_seconds] [--with-checkpoints]}"
RUN_NAME="${2:?Usage: $0 <node_ip> <run_name> [interval_seconds] [--with-checkpoints]}"
INTERVAL="${3:-0}"
WITH_CHECKPOINTS=0
if [ "$4" = "--with-checkpoints" ] || [ "$3" = "--with-checkpoints" ]; then
    WITH_CHECKPOINTS=1
    [ "$3" = "--with-checkpoints" ] && INTERVAL=0
fi
STALE_TIMEOUT=3600  # 1 hour with no changes = exit

REMOTE_DIR="ubuntu@${NODE_IP}:~/results/${RUN_NAME}/"
LOCAL_DIR="/efs/rlvr-experiments/results/${RUN_NAME}/"

LAST_CHANGE=$(date +%s)

sync_once() {
    CHANGED=0
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Syncing from ${NODE_IP}:${RUN_NAME}..."

    # Create local dir
    mkdir -p "$LOCAL_DIR"

    # Rsync with checksum to detect changes
    # --info=progress2 shows overall progress
    if [ "$WITH_CHECKPOINTS" -eq 1 ]; then
        # Include model weights
        OUTPUT=$(rsync -avz --info=stats2 "$REMOTE_DIR" "$LOCAL_DIR" 2>&1)
    else
        # Exclude large model checkpoints (sync separately if needed)
        OUTPUT=$(rsync -avz --info=stats2 \
            --exclude='*.safetensors' \
            --exclude='*.bin' \
            --exclude='*.pt' \
            "$REMOTE_DIR" "$LOCAL_DIR" 2>&1)
    fi

    # Check if anything was transferred
    if echo "$OUTPUT" | grep -q "Number of regular files transferred: [1-9]"; then
        CHANGED=1
        echo "$OUTPUT" | grep -E "(transferred|total size)"
    else
        echo "  No changes."
    fi
}

# Main loop
echo "Syncing: ${NODE_IP}:~/results/${RUN_NAME}/ -> ${LOCAL_DIR}"

if [ "$INTERVAL" -gt 0 ]; then
    echo "Polling every ${INTERVAL}s. Auto-exits after ${STALE_TIMEOUT}s with no changes."
    while true; do
        sync_once
        if [ "$CHANGED" -eq 1 ]; then
            LAST_CHANGE=$(date +%s)
        fi

        # Check for stale timeout
        NOW=$(date +%s)
        ELAPSED=$((NOW - LAST_CHANGE))
        if [ "$ELAPSED" -ge "$STALE_TIMEOUT" ]; then
            echo "No changes for ${STALE_TIMEOUT}s. Exiting."
            exit 0
        fi

        sleep "$INTERVAL"
    done
else
    sync_once
fi
