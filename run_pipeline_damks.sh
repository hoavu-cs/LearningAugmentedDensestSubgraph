#!/bin/bash
# ---------------------------------------------------------------------------
# Pipeline: generate labeled DamkS dataset → run DamkS experiment
# Usage:   ./run_pipeline_damks.sh [INPUT_JSON] [K]
# Example: ./run_pipeline_damks.sh data/twitch_edges.json 15
# ---------------------------------------------------------------------------

INPUT_FILE="${1:-data/twitch_edges.json}"
K="${2:-15}"

BASE="${INPUT_FILE%.json}"
TRAIN_FILE="${BASE}_train.json"
VAL_FILE="${BASE}_val.json"

echo "============================================================"
echo "DamkS Pipeline config"
echo "  Input   : $INPUT_FILE"
echo "  Train   : $TRAIN_FILE"
echo "  Val     : $VAL_FILE"
echo "  K       : $K"
echo "============================================================"

set -e

echo ""
echo ">>> Step 1: create labeled DamkS dataset"
julia --project=. --threads auto create_labeled_dataset_damks.jl "$INPUT_FILE" "$K"

echo ""
echo ">>> Step 2: run DamkS experiment"
julia --project=. --threads auto damks_experiment.jl "$TRAIN_FILE" "$VAL_FILE" "$K"
