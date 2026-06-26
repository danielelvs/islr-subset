#!/usr/bin/env bash
set -euo pipefail

mkdir -p scripts/batch/ksl scripts/batch/include50 scripts/setup scripts/preprocessing scripts/analysis scripts/utils

move_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    echo "moved: $src -> $dst"
  fi
}

move_if_exists scripts/run_ksl_batch.py scripts/batch/ksl/run_batch.py
move_if_exists scripts/run_ksl_all.sh scripts/batch/ksl/run_all.sh
move_if_exists scripts/summarize_ksl_results.py scripts/batch/ksl/summarize_results.py
move_if_exists scripts/run_include50_batch.py scripts/batch/include50/run_batch.py
move_if_exists scripts/run_include50_all.sh scripts/batch/include50/run_all.sh
move_if_exists scripts/run_include50_imputation_only.sh scripts/batch/include50/run_imputation_only.sh
move_if_exists scripts/summarize_include50_results.py scripts/batch/include50/summarize_results.py

echo "Estrutura criada/organizada. Confira scripts/batch/."
