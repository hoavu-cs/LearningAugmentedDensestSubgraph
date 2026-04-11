#!/bin/bash
# ---------------------------------------------------------------------------
# Pipeline: generate labeled DS dataset → run DS experiment
# Usage:   ./run_pipeline_ds.sh [INPUT_JSON] [MAX_NODES]
# Example: ./run_pipeline_ds.sh data/twitch_edges.json 100
# ---------------------------------------------------------------------------

INPUT_FILE="${1:-data/twitch_edges.json}"
MAX_NODES="${2:-100}"

BASE="${INPUT_FILE%.json}"
TRAIN_FILE="${BASE}_train.json"
VAL_FILE="${BASE}_val.json"

echo "============================================================"
echo "DS Pipeline config"
echo "  Input   : $INPUT_FILE"
echo "  Train   : $TRAIN_FILE"
echo "  Val     : $VAL_FILE"
echo "  MaxNodes: $MAX_NODES"
echo "============================================================"

set -e

echo ""
echo ">>> Step 1: create labeled DS dataset"
julia --project=. --threads auto create_labeled_dataset_ds.jl "$INPUT_FILE" "$MAX_NODES"

echo ""
echo ">>> Step 2: run DS experiment"
julia --project=. --threads auto ds_experiment.jl "$TRAIN_FILE" "$VAL_FILE"
