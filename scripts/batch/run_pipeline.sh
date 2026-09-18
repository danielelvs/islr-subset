#!/usr/bin/env bash
set -euo pipefail

# Run the configured experiment grid for one dataset or for all datasets.
# "all" means four independent benchmarks with their proper protocols; it does
# not concatenate label spaces or train one model over the four vocabularies.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/batch/run_pipeline.sh <all|ksl|minds|ufop|include50> [--dry-run]

Examples:
  ./scripts/batch/run_pipeline.sh ksl --dry-run
  ./scripts/batch/run_pipeline.sh ksl
  ./scripts/batch/run_pipeline.sh all

The dataset configuration controls subsets, imputation, epochs and device:
  configs/datasets/<dataset>.env

--dry-run validates configurations, input files and CSV headers without training.
EOF
}

TARGET="${1:-}"
DRY_RUN="false"
if [[ -z "$TARGET" ]]; then
  usage
  exit 2
fi
if [[ "$TARGET" == "-h" || "$TARGET" == "--help" ]]; then
  usage
  exit 0
fi
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN="true" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case "$TARGET" in
  all) DATASETS=(include50 ksl minds ufop) ;;
  include50|ksl|minds|ufop) DATASETS=("$TARGET") ;;
  *) echo "Unknown dataset: $TARGET" >&2; usage >&2; exit 2 ;;
esac

validate_dataset() {
  local name="$1"
  local config_file="configs/datasets/${name}.env"
  if [[ ! -f "$config_file" ]]; then
    echo "Configuration file not found: $config_file" >&2
    return 1
  fi

  # Load each config in a subshell so values cannot leak to another dataset.
  (
    set -u
    # shellcheck source=/dev/null
    source "$config_file"
    : "${DATASET:?DATASET is missing in $config_file}"
    : "${DATA_CSV:?DATA_CSV is missing in $config_file}"
    : "${RESULTS_DIR:?RESULTS_DIR is missing in $config_file}"
    : "${SUBSETS:?SUBSETS is missing in $config_file}"
    : "${IMPUTATIONS:?IMPUTATIONS is missing in $config_file}"
    : "${PROTOCOL:?PROTOCOL is missing in $config_file}"

    if [[ ! -f "$DATA_CSV" ]]; then
      echo "Input CSV not found: $DATA_CSV" >&2
      exit 1
    fi
    if [[ "$name" == "include50" && "$PROTOCOL" != "fixed_split" ]]; then
      echo "include50 must use PROTOCOL=fixed_split, found: $PROTOCOL" >&2
      exit 1
    fi
    if [[ "$name" != "include50" && "$PROTOCOL" != "nested_lopo" && "$PROTOCOL" != "lopo_fixed_val" ]]; then
      echo "$name must use a LOPO protocol, found: $PROTOCOL" >&2
      exit 1
    fi

    echo "[$name] protocol=$PROTOCOL"
    echo "  input=$DATA_CSV"
    echo "  results=$RESULTS_DIR"
    echo "  subsets=$SUBSETS"
    echo "  imputations=$IMPUTATIONS"
    validation_args=(
      --data-csv "$DATA_CSV"
      --nrows 100
    )
    [[ -n "${PERSON_COL:-}" ]] && validation_args+=(--person-col "$PERSON_COL")
    [[ -n "${CATEGORY_COL:-}" ]] && validation_args+=(--category-col "$CATEGORY_COL")
    [[ -n "${VIDEO_COL:-}" ]] && validation_args+=(--video-col "$VIDEO_COL")
    [[ -n "${FRAME_COL:-}" ]] && validation_args+=(--frame-col "$FRAME_COL")
    python scripts/preprocessing/validate_dataset_csv.py "${validation_args[@]}"
  )
}

echo "Preflight: ${DATASETS[*]}"
for dataset_name in "${DATASETS[@]}"; do
  validate_dataset "$dataset_name"
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run completed. No training was started."
  exit 0
fi

for dataset_name in "${DATASETS[@]}"; do
  echo "Starting experiment grid: $dataset_name"
  if [[ "$dataset_name" == "include50" ]]; then
    ./scripts/batch/run_include50_split_all.sh
  else
    ./scripts/batch/run_dataset_all.sh "$dataset_name"
  fi
done

echo "Requested experiment grids completed."
