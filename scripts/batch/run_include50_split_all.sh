#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="configs/datasets/include50.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE"
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${DATA_CSV:?DATA_CSV is not defined in the configuration file}"
: "${RESULTS_DIR:?RESULTS_DIR is not defined in the configuration file}"
: "${SUBSETS:?SUBSETS is not defined in the configuration file}"
: "${IMPUTATIONS:?IMPUTATIONS is not defined in the configuration file}"

EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-0.0001}"
WD="${WD:-0.0001}"
DEVICE="${DEVICE:-auto}"
SAVE_MODELS="${SAVE_MODELS:-false}"
CACHE_PREPROCESSED="${CACHE_PREPROCESSED:-false}"

mkdir -p "$RESULTS_DIR" logs
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

for subset in $SUBSETS; do
  for imputation in $IMPUTATIONS; do
    echo
    echo "============================================================"
    echo "INCLUDE-50 | subset=${subset} | imputation=${imputation}"
    echo "============================================================"

    python scripts/batch/run_include50_split.py \
      --dataset include50 \
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
      --save-model "$SAVE_MODELS" \
      --cache-preprocessed "$CACHE_PREPROCESSED" \
      --resume true \
      2>&1 | tee "logs/include50_${subset}_imputation-${imputation}.log"
  done
done
