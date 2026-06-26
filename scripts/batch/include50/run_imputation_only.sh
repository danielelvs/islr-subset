#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

source venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p logs
mkdir -p experiments/include50_nested_lopo_resume_grid
mkdir -p reports/include50

DATA_CSV="data/interim/include50/include50_mediapipe.csv"
RESULTS_DIR="experiments/include50_nested_lopo_resume_grid"

# Versão mais curta: roda somente com imputação, que é a condição principal no artigo.
for subset in 2nd arcanjo laines 1st all; do
  imputation=true
  echo "=========================================="
  echo "Running INCLUDE-50 subset=${subset} imputation=${imputation}"
  echo "=========================================="

  python scripts/batch/include50/run_batch.py     --dataset include50     --data-csv "$DATA_CSV"     --results-dir "$RESULTS_DIR"     --subset "$subset"     --imputation "$imputation"     --max-runs none     --device cuda     2>&1 | tee "logs/include50_${subset}_imputation-${imputation}.log"
done
