#!/usr/bin/env bash
# ==============================================================================
# Repeatable HIL & Sensor Validation Script for ROS 2 Edge Perception
# ==============================================================================
set -euo pipefail

HARDWARE="host"
DATASET="warehouse_01"
DURATION=30
FAULT_INJECTION="--fault-injection"
OUTPUT_REPORT="validation_report.json"

while [[ $# -gt 0 ]]; do
  case $1 in
    --hardware)
      HARDWARE="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --output)
      OUTPUT_REPORT="$2"
      shift 2
      ;;
    --no-fault-injection)
      FAULT_INJECTION="--no-fault-injection"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "======================================================================"
echo " Starting HIL Validation Run: $HARDWARE | Dataset: $DATASET | ${DURATION}s"
echo "======================================================================"

python3 scripts/run_validation.py \
  --hardware "$HARDWARE" \
  --dataset "$DATASET" \
  --duration "$DURATION" \
  $FAULT_INJECTION \
  --output "$OUTPUT_REPORT"
