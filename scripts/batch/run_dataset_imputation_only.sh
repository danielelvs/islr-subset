#!/usr/bin/env bash
set -euo pipefail

# Same as run_dataset_all.sh, but forces only imputation=true.
# Usage:
#   ./scripts/batch/run_dataset_imputation_only.sh ksl

DATASET_NAME="${1:-}"
if [[ -z "$DATASET_NAME" ]]; then
  echo "Uso: $0 <dataset>"
  exit 1
fi

CONFIG_FILE="configs/datasets/${DATASET_NAME}.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config não encontrada: $CONFIG_FILE"
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"
export IMPUTATIONS="true"

# Run inline instead of recursively editing config.
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.0001}"
WD="${WD:-0.0001}"
DEVICE="${DEVICE:-cuda}"
MAX_RUNS="${MAX_RUNS:-none}"
PROTOCOL="${PROTOCOL:-nested_lopo}"

mkdir -p logs "$RESULTS_DIR"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

for subset in $SUBSETS; do
  echo "=========================================="
  echo "Dataset=${DATASET} subset=${subset} imputation=true"
  echo "Protocol=${PROTOCOL} epochs=${EPOCHS} patience=${PATIENCE}"
  echo "=========================================="

  python scripts/batch/run_dataset_batch.py \
    --dataset "$DATASET" \
    --data-csv "$DATA_CSV" \
    --results-dir "$RESULTS_DIR" \
    --subset "$subset" \
    --imputation true \
    --max-runs "$MAX_RUNS" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --wd "$WD" \
    --protocol "$PROTOCOL" \
    ${PERSON_COL:+--person-col "$PERSON_COL"} \
    ${CATEGORY_COL:+--category-col "$CATEGORY_COL"} \
    ${VIDEO_COL:+--video-col "$VIDEO_COL"} \
    ${FRAME_COL:+--frame-col "$FRAME_COL"} \
    2>&1 | tee "logs/${DATASET}_${subset}_imputation-true.log"
done
