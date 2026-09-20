#!/usr/bin/env bash
# ==============================================================================
# 1-Click Launch Script for ROS 2 Edge Perception (Linux / Cloud Shell / WSL)
# ==============================================================================
set -e

echo "======================================================================"
echo " Starting Commercial-Grade ROS 2 Edge Perception Pipeline"
echo "======================================================================"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not in PATH."
    exit 1
fi

MODE="${1:-cpu}"

if [ "$MODE" == "rocm" ]; then
    echo "[INFO] Running in AMD ROCm GPU Mode..."
    docker compose up --build perception-rocm
else
    echo "[INFO] Running in CPU Mode (Zero-cost, works on any PC / Cloud Shell)..."
    docker compose up --build perception-cpu
fi

