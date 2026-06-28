#!/usr/bin/env bash
set -euo pipefail

# Generic wrapper for running all subset/imputation conditions for a dataset.
# Usage:
#   ./scripts/batch/run_dataset_all.sh ksl
#   ./scripts/batch/run_dataset_all.sh include50
#   ./scripts/batch/run_dataset_all.sh minds
#   ./scripts/batch/run_dataset_all.sh ufop
#
# It loads configs/datasets/<dataset>.env.

DATASET_NAME="${1:-}"
if [[ -z "$DATASET_NAME" ]]; then
  echo "Uso: $0 <dataset>"
  echo "Exemplos: $0 ksl | $0 include50 | $0 minds | $0 ufop"
  exit 1
fi

CONFIG_FILE="configs/datasets/${DATASET_NAME}.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config não encontrada: $CONFIG_FILE"
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${DATASET:?DATASET não definido no config}"
: "${DATA_CSV:?DATA_CSV não definido no config}"
: "${RESULTS_DIR:?RESULTS_DIR não definido no config}"
: "${SUBSETS:?SUBSETS não definido no config}"
: "${IMPUTATIONS:?IMPUTATIONS não definido no config}"

EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.0001}"
WD="${WD:-0.0001}"
DEVICE="${DEVICE:-cuda}"
MAX_RUNS="${MAX_RUNS:-none}"
PROTOCOL="${PROTOCOL:-nested_lopo}"

mkdir -p logs "$RESULTS_DIR"

# Ensure local src is importable.
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

for subset in $SUBSETS; do
  for imputation in $IMPUTATIONS; do
    echo "=========================================="
    echo "Dataset=${DATASET} subset=${subset} imputation=${imputation}"
    echo "Protocol=${PROTOCOL} epochs=${EPOCHS} patience=${PATIENCE}"
    echo "=========================================="

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
      --lr "$LR" \
      --wd "$WD" \
      --protocol "$PROTOCOL" \
      ${PERSON_COL:+--person-col "$PERSON_COL"} \
      ${CATEGORY_COL:+--category-col "$CATEGORY_COL"} \
      ${VIDEO_COL:+--video-col "$VIDEO_COL"} \
      ${FRAME_COL:+--frame-col "$FRAME_COL"} \
      2>&1 | tee "logs/${DATASET}_${subset}_imputation-${imputation}.log"
  done
done
