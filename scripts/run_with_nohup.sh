#!/bin/bash
#
# run_with_nohup.sh — Run long jobs with nohup, auto-detecting available GPU
#
# Usage:
#   ./scripts/run_with_nohup.sh <command> [args...] [--name <job_name>]
#
# Examples:
#   ./scripts/run_with_nohup.sh python pipeline/preprocessing.py --config configs/config_rtx.yaml --workers 8
#   ./scripts/run_with_nohup.sh python pipeline/train.py --config configs/config_rtx.yaml --fold all --track A --name train_all_folds
#   ./scripts/run_with_nohup.sh python pipeline/splits.py --config configs/config_rtx.yaml --inspect --name gen_splits
#
# Features:
#   - Auto-detects available CUDA GPU
#   - Creates log directory at /data/derrick/isles26/logs/
nohup
#   - Saves stdout/stderr to logs/<job_name>.log
#   - Prints command to run locally (without nohup) for testing
#   - Shows command to check job status
#

set -e

# Parse arguments
JOB_NAME=""
CMD_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            JOB_NAME="$2"
            shift 2
            ;;
        *)
            CMD_ARGS+=("$1")
            shift
            ;;
    esac
done

# Generate default job name from command if not provided
if [ -z "$JOB_NAME" ]; then
    # Extract base command name
    BASE_CMD="${CMD_ARGS[0]}"
    if [[ "$BASE_CMD" == *"python"* ]]; then
        # Get script name without path and extension
        SCRIPT_NAME=$(basename "${CMD_ARGS[1]}")
        JOB_NAME="job_${SCRIPT_NAME%.*}"
    else
        JOB_NAME="job_$(echo "$BASE_CMD" | tr ' ' '_')"
    fi
fi

# Auto-detect available GPU
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        # Check for available GPU with memory free
        local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null || echo "0")

        if [ "$gpu_count" -gt 0 ]; then
            # Get first GPU with free memory
            local free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)

            if [ -n "$free_memory" ] && [ "$free_memory" -gt 1000 ]; then
                GPU_ID=0
                echo "Using GPU $GPU_ID (available memory: ${free_memory}MB)"
                return 0
            fi
        fi

        # No GPU available, set to CPU
        GPU_ID="-1"
        echo "No GPU available, running in CPU mode"
        return 0
    else
        GPU_ID="-1"
        echo "nvidia-smi not found, running in CPU mode"
        return 0
    fi
}

# Detect GPU before running
GPU_ID=""
detect_gpu

# Set CUDA visible device
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Define log directory
LOG_DIR="/data/derrick/isles26/logs"
LOG_FILE="$LOG_DIR/${JOB_NAME}.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Print summary
echo ""
echo "=== Job Configuration ==="
echo "Job name:    $JOB_NAME"
echo "GPU visible: $GPU_ID"
echo "Log file:    $LOG_FILE"
echo ""
echo "Command to run:"
echo "  ${CMD_ARGS[*]}"
echo ""
echo "=== Running with nohup ==="
echo "To monitor: tail -f $LOG_FILE"
echo "To check status: ps aux | grep $JOB_NAME"
echo ""

# Check if command is python
if [[ "${CMD_ARGS[0]}" == "python" ]]; then
    # For Python scripts, also print the nohup command to run locally
    echo "=== To run locally (without nohup): ==="
    echo "CUDA_VISIBLE_DEVICES=$GPU_ID ${CMD_ARGS[*]}"
    echo ""
fi

# Run the command with nohup
nohup "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1 &
JOB_PID=$!

echo "Job started with PID: $JOB_PID"
echo "Output saved to: $LOG_FILE"
