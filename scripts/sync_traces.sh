#!/bin/bash
# Sync trace.jsonl files from S3 SageMaker jobs to local results directory
# Usage: ./scripts/sync_traces.sh [job_pattern] [interval_seconds]
#
# Examples:
#   ./scripts/sync_traces.sh                    # Sync all jobs from today, once
#   ./scripts/sync_traces.sh 192                # Sync jobs matching pattern
#   ./scripts/sync_traces.sh 192 60             # Poll every 60s

set -e

JOB_PATTERN="${1:-}"
INTERVAL="${2:-0}"
STALE_TIMEOUT=3600  # 1 hour with no changes = exit

S3_BASE="s3://sagemaker-us-west-2-503561457547/rlvr-experiments/checkpoints"
LOCAL_BASE="results"

LAST_CHANGE=$(date +%s)

sync_once() {
    CHANGED=0
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Syncing traces..."

    # List jobs from S3
    if [ -n "$JOB_PATTERN" ]; then
        JOBS=$(aws s3 ls "$S3_BASE/" | awk '{print $2}' | tr -d '/' | grep "$JOB_PATTERN" || true)
    else
        # Default: get jobs from today
        TODAY=$(date +%Y%m%d)
        JOBS=$(aws s3 ls "$S3_BASE/" | awk '{print $2}' | tr -d '/' | grep "$TODAY" || true)
    fi

    if [ -z "$JOBS" ]; then
        echo "  No jobs found matching pattern"
        return
    fi

    for JOB in $JOBS; do
        # Check if traces folder exists
        TRACE_FILE="$S3_BASE/$JOB/traces/trace.jsonl"
        S3_SIZE=$(aws s3 ls "$TRACE_FILE" 2>/dev/null | awk '{print $3}' || echo "0")

        if [ "$S3_SIZE" = "0" ]; then
            continue
        fi

        # Create local dir and check local size
        LOCAL_DIR="$LOCAL_BASE/$JOB/traces"
        LOCAL_FILE="$LOCAL_DIR/trace.jsonl"
        mkdir -p "$LOCAL_DIR"

        LOCAL_SIZE=$(stat -c%s "$LOCAL_FILE" 2>/dev/null || stat -f%z "$LOCAL_FILE" 2>/dev/null || echo "0")

        if [ "$S3_SIZE" != "$LOCAL_SIZE" ]; then
            echo "  $JOB: $((S3_SIZE/1024/1024))MB (was $((LOCAL_SIZE/1024/1024))MB)"
            aws s3 cp "$TRACE_FILE" "$LOCAL_FILE" --quiet
            CHANGED=1
        fi

        # Also grab config.yaml if we don't have it yet
        CONFIG_FILE="$LOCAL_DIR/../config.yaml"
        if [ ! -f "$CONFIG_FILE" ]; then
            S3_CONFIG="$S3_BASE/$JOB/config.yaml"
            if aws s3 ls "$S3_CONFIG" &>/dev/null; then
                aws s3 cp "$S3_CONFIG" "$CONFIG_FILE" --quiet
                echo "  $JOB: fetched config.yaml"
            fi
        fi
    done

    echo "  Done."
}

# Main loop
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
