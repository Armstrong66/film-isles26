#!/bin/bash
#
# run_job.sh — Run ISLES26 jobs with GPU detection and nohup support
#
# Usage:
#   ./scripts/run_job.sh <job_type> [options]
#
# Job types:
#   preprocess  - Run preprocessing pipeline
#   splits      - Generate CV splits
#   train       - Train model
#   evaluate    - Evaluate model with TTA
#
# Options:
#   --name <job_name>   Custom job name for log files
#   --fold <n|all>      Training fold (default: all)
#   --track <A|C>       Training track (default: A)
#   --workers <n>       Number of workers (default: 8)
#   --local             Run locally without nohup (for testing)
#
# Examples:
#   ./scripts/run_job.sh preprocess --name preproc_rtx
#   ./scripts/run_job.sh train --fold all --track A --name train_all_folds
#   ./scripts/run_job.sh evaluate --fold all --tta --name eval_all_tta
#   ./scripts/run_job.sh train --fold 0 --local --name train_fold0_test
#

set -e

# Default values
JOB_NAME=""
JOB_TYPE=""
CMD_ARGS=("--config" "configs/config_rtx.yaml")
LOCAL_MODE=false
LOG_DIR="/data/derrick/isles26/logs"

# GPU detection function - finds the least busy GPU
# Always returns a GPU ID (0 or 1), never CPU mode for training
detect_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "0"
        return 0
    fi

    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null || echo "0")

    # If no GPUs or count is 0, default to GPU 0
    if [ -z "$gpu_count" ] || [ "$gpu_count" -eq 0 ]; then
        echo "0"
        return 0
    fi

    # For RTX with 2 GPUs, find the least busy one
    if [ "$gpu_count" -ge 2 ]; then
        local gpu0_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n '1p')
        local gpu1_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n '2p')
        local gpu0_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed -n '1p')
        local gpu1_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed -n '2p')

        # Default to GPU 0 if any query fails
        if [ -z "$gpu0_mem" ] || [ -z "$gpu1_mem" ]; then
            echo "0"
            return 0
        fi

        # Compare total load (memory + utilization)
        # Lower total score = less busy
        local gpu0_load=$((gpu0_mem + gpu0_util))
        local gpu1_load=$((gpu1_mem + gpu1_util))

        if [ "$gpu1_load" -lt "$gpu0_load" ]; then
            echo "1"
        else
            echo "0"
        fi
    else
        # Single GPU system
        echo "0"
    fi
    return 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        preprocess|splits|train|evaluate)
            JOB_TYPE="$1"
            shift
            ;;
        --name)
            JOB_NAME="$2"
            shift 2
            ;;
        --fold)
            CMD_ARGS+=("--fold" "$2")
            shift 2
            ;;
        --track)
            CMD_ARGS+=("--track" "$2")
            shift 2
            ;;
        --workers)
            CMD_ARGS+=("--workers" "$2")
            shift 2
            ;;
        --tta)
            CMD_ARGS+=("--tta")
            shift
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        *)
            CMD_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate job type
if [ -z "$JOB_TYPE" ]; then
    echo "Error: Job type required (preprocess, splits, train, evaluate)"
    echo ""
    echo "Usage: $0 <job_type> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 preprocess --name preproc_rtx"
    echo "  $0 train --fold all --track A --name train_all_folds"
    echo "  $0 evaluate --fold all --tta"
    exit 1
fi

# Generate default job name
if [ -z "$JOB_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    FOLD_ARG=$(printf "%s" "${CMD_ARGS[@]}" | grep -o "fold [0-9]*" | head -1 | sed 's/fold //')
    TRACK_ARG=$(printf "%s" "${CMD_ARGS[@]}" | grep -o "track [A-C]" | head -1 | sed 's/track //')
    FOLD_STR="${FOLD_ARG:-all}"
    TRACK_STR="${TRACK_ARG:-A}"
    JOB_NAME="${JOB_TYPE}_${FOLD_STR}folds_${TRACK_STR}"
fi

# Set CUDA_VISIBLE_DEVICES
GPU_ID=$(detect_gpu)
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Log file setup
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}.log"

# Build the Python command
PYTHON_CMD="python pipeline/${JOB_TYPE}.py"

# Print summary
echo ""
echo "=== Job Configuration ==="
echo "Job type:    $JOB_TYPE"
echo "Job name:    $JOB_NAME"
echo "GPU visible: $GPU_ID"
echo "Log file:    $LOG_FILE"
echo ""
echo "Command:"
echo "  $PYTHON_CMD ${CMD_ARGS[*]}"
echo ""

# Check if running locally
if [ "$LOCAL_MODE" = true ]; then
    echo "=== Running locally (no nohup) ==="
    echo "Press Ctrl+C to stop"
    echo ""
    $PYTHON_CMD "${CMD_ARGS[@]}"
else
    echo "=== Running with nohup ==="
    echo "To monitor: tail -f $LOG_FILE"
    echo "To check status: ps aux | grep ${JOB_NAME}"
    echo ""

    # Run with nohup
    nohup $PYTHON_CMD "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    JOB_PID=$!

    echo "Job started with PID: $JOB_PID"
    echo "Output saved to: $LOG_FILE"

    # Show command to reproduce locally
    echo ""
    echo "=== To run locally (without nohup): ==="
    echo "CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_CMD ${CMD_ARGS[*]}"
fi
