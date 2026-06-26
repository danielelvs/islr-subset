#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

source venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p logs
mkdir -p experiments/ksl_nested_lopo_resume_grid
mkdir -p reports/ksl

DATA_CSV="data/interim/ksl/ksl_mediapipe.csv"
RESULTS_DIR="experiments/ksl_nested_lopo_resume_grid"


for subset in 2nd arcanjo laines 1st all; do
  for imputation in true false; do
    echo "=========================================="
    echo "Running KSL subset=${subset} imputation=${imputation}"
    echo "=========================================="

    python scripts/batch/ksl/run_batch.py       --dataset ksl       --data-csv "$DATA_CSV"       --results-dir "$RESULTS_DIR"       --subset "$subset"       --imputation "$imputation"       --max-runs none       --device cuda       2>&1 | tee "logs/ksl_${subset}_imputation-${imputation}.log"
  done
done
