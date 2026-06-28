#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="experiments/include50_fixed_split_grid"
DATA_CSV="data/interim/include50/include50_mediapipe_with_split.csv"

mkdir -p "$RESULTS_DIR"

SUBSETS=("all" "laines" "arcanjo" "1st" "2nd")
IMPUTATIONS=("true" "false")

for SUBSET in "${SUBSETS[@]}"; do
  for IMPUTATION in "${IMPUTATIONS[@]}"; do
    echo
    echo "========================================"
    echo "INCLUDE-50 | subset=${SUBSET} | imputation=${IMPUTATION}"
    echo "========================================"

    python scripts/batch/run_include50_fixed_split.py \
      --dataset include50 \
      --data-csv "$DATA_CSV" \
      --results-dir "$RESULTS_DIR" \
      --subset "$SUBSET" \
      --imputation "$IMPUTATION" \
      --device cuda \
      --epochs 30 \
      --batch-size 64 \
      --lr 1e-4 \
      --wd 1e-4 \
      --patience 5 \
      --resume true
  done
done
