#!/usr/bin/env bash
set -euo pipefail

DATA_CSV="${DATA_CSV:-data/interim/ksl/ksl_mediapipe.csv}"
RESULTS_DIR="${RESULTS_DIR:-experiments/ksl_alves_groups_16_4}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-1e-4}"
WD="${WD:-1e-4}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-1638102311}"

SAVE_MODEL="${SAVE_MODEL:-false}"
CACHE_PREPROCESSED="${CACHE_PREPROCESSED:-false}"
RESUME="${RESUME:-true}"

SUBSETS=(all laines arcanjo 1st 2nd)
IMPUTATIONS=(false true)

echo "============================================================"
echo "KSL Alves signer-group reproduction experiment"
echo "5 subsets x 2 imputation settings x 5 folds = 50 runs"
echo "Data:    $DATA_CSV"
echo "Results: $RESULTS_DIR"
echo "Device:  $DEVICE"
echo "============================================================"

for subset in "${SUBSETS[@]}"; do
  for imputation in "${IMPUTATIONS[@]}"; do
    echo
    echo "------------------------------------------------------------"
    echo "subset=$subset | imputation=$imputation"
    echo "------------------------------------------------------------"

    python -u scripts/batch/run_ksl_grouped_12_4_4.py \
      --data-csv "$DATA_CSV" \
      --results-dir "$RESULTS_DIR" \
      --subset "$subset" \
      --imputation "$imputation" \
      --device "$DEVICE" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --lr "$LR" \
      --wd "$WD" \
      --patience "$PATIENCE" \
      --seed "$SEED" \
      --save-model "$SAVE_MODEL" \
      --cache-preprocessed "$CACHE_PREPROCESSED" \
      --resume "$RESUME"
  done
done

echo
echo "All Alves signer-group reproduction conditions finished."
