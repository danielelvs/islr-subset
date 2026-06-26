# ISLR scripts organizados

Estrutura sugerida:

```text
scripts/
├── batch/
│   ├── ksl/
│   │   ├── run_batch.py
│   │   ├── run_all.sh
│   │   ├── run_imputation_only.sh
│   │   └── summarize_results.py
│   └── include50/
│       ├── run_batch.py
│       ├── run_all.sh
│       ├── run_imputation_only.sh
│       └── summarize_results.py
├── setup/
│   ├── check_environment.py
│   └── organize_legacy_batch_scripts.sh
├── preprocessing/
│   └── validate_dataset_csv.py
├── analysis/
│   └── summarize_all_datasets.py
└── utils/
    ├── check_gpu_usage.py
    ├── count_result_files.py
    └── inspect_result_json.py
```

## Instalação

Extraia o ZIP na raiz do projeto. Depois:

```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m pip install -r requirements-batch-py38.txt
```

## Check do ambiente

```bash
python scripts/setup/check_environment.py
```

## Validar CSV

```bash
python scripts/preprocessing/validate_dataset_csv.py data/interim/ksl/ksl_mediapipe.csv
python scripts/preprocessing/validate_dataset_csv.py data/interim/include50/include50_mediapipe.csv
```

## KSL

Teste pequeno:

```bash
python scripts/batch/ksl/run_batch.py   --dataset ksl   --data-csv data/interim/ksl/ksl_mediapipe.csv   --results-dir experiments/ksl_nested_lopo_resume_grid   --subset 2nd   --imputation true   --max-runs 2   --device cuda
```

Rodar tudo:

```bash
chmod +x scripts/batch/ksl/run_all.sh
./scripts/batch/ksl/run_all.sh
```

Somente com imputação:

```bash
chmod +x scripts/batch/ksl/run_imputation_only.sh
./scripts/batch/ksl/run_imputation_only.sh
```

Resumo:

```bash
python scripts/batch/ksl/summarize_results.py   --results-dir experiments/ksl_nested_lopo_resume_grid   --reports-dir reports/ksl
```

## INCLUDE-50

Teste pequeno:

```bash
python scripts/batch/include50/run_batch.py   --dataset include50   --data-csv data/interim/include50/include50_mediapipe.csv   --results-dir experiments/include50_nested_lopo_resume_grid   --subset 2nd   --imputation true   --max-runs 2   --device cuda
```

Rodar tudo:

```bash
chmod +x scripts/batch/include50/run_all.sh
./scripts/batch/include50/run_all.sh
```

Somente com imputação:

```bash
chmod +x scripts/batch/include50/run_imputation_only.sh
./scripts/batch/include50/run_imputation_only.sh
```

Resumo:

```bash
python scripts/batch/include50/summarize_results.py   --results-dir experiments/include50_nested_lopo_resume_grid   --reports-dir reports/include50
```

## Utilitários

Contar resultados:

```bash
python scripts/utils/count_result_files.py experiments/ksl_nested_lopo_resume_grid
```

Ver GPU:

```bash
python scripts/utils/check_gpu_usage.py
```

Combinar summaries:

```bash
python scripts/analysis/summarize_all_datasets.py
```
