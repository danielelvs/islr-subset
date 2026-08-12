#!/usr/bin/env bash
set -euo pipefail

# Run every configured subset/imputation condition for a LOPO dataset.
# Usage: ./scripts/batch/run_dataset_all.sh ksl

DATASET_NAME="${1:-}"
if [[ -z "$DATASET_NAME" ]]; then
  echo "Usage: $0 <dataset>"
  echo "Examples: $0 ksl | $0 minds | $0 ufop"
  exit 1
fi

if [[ "$DATASET_NAME" == "include50" ]]; then
  echo "INCLUDE-50 uses a fixed train/validation/test split."
  echo "Run: ./scripts/batch/run_include50_split_all.sh"
  exit 1
fi

CONFIG_FILE="configs/datasets/${DATASET_NAME}.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE"
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${DATASET:?DATASET is not defined in the configuration file}"
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
DEVICE="${DEVICE:-cuda}"
MAX_RUNS="${MAX_RUNS:-none}"
PROTOCOL="${PROTOCOL:-nested_lopo}"
SAVE_MODELS="${SAVE_MODELS:-false}"
CACHE_PREPROCESSED="${CACHE_PREPROCESSED:-false}"

mkdir -p logs "$RESULTS_DIR"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

for subset in $SUBSETS; do
  for imputation in $IMPUTATIONS; do
    echo "============================================================"
    echo "Dataset=${DATASET} subset=${subset} imputation=${imputation}"
    echo "Protocol=${PROTOCOL} epochs=${EPOCHS} patience=${PATIENCE}"
    echo "============================================================"

    python scripts/batch/run_dataset_batch.py \
      --dataset "$DATASET" \
      --data-csv "$DATA_CSV" \
      --results-dir "$RESULTS_DIR" \
      --subset "$subset" \
      --imputation "$imputation" \
      --max-runs "$MAX_RUNS" \
      --device "$DEVICE" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --lr "$LR" \
      --wd "$WD" \
      --protocol "$PROTOCOL" \
      --save-model "$SAVE_MODELS" \
      --cache-preprocessed "$CACHE_PREPROCESSED" \
      ${PERSON_COL:+--person-col "$PERSON_COL"} \
      ${CATEGORY_COL:+--category-col "$CATEGORY_COL"} \
      ${VIDEO_COL:+--video-col "$VIDEO_COL"} \
      ${FRAME_COL:+--frame-col "$FRAME_COL"} \
      2>&1 | tee "logs/${DATASET}_${subset}_imputation-${imputation}.log"
  done
done
