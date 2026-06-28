# ISLR Generic Dataset Batch Scripts

Este bundle reorganiza os scripts para rodar **qualquer dataset** que esteja no mesmo formato geral de CSV de landmarks MediaPipe.

Exemplos já configurados:

- `ksl`
- `include50`
- `minds`
- `ufop`

A ideia é usar **um batch Python genérico** e trocar apenas o arquivo de configuração do dataset.

---

## Estrutura

```text
scripts/
├── batch/
│   ├── run_dataset_batch.py
│   ├── run_dataset_all.sh
│   ├── run_dataset_imputation_only.sh
│   └── summarize_dataset_results.py
├── setup/
│   └── check_environment.py
├── preprocessing/
│   └── validate_dataset_csv.py
├── analysis/
│   └── summarize_all_datasets.py
└── utils/
    ├── check_gpu_usage.py
    ├── count_result_files.py
    └── inspect_result_json.py

configs/
└── datasets/
    ├── ksl.env
    ├── include50.env
    ├── minds.env
    └── ufop.env
```

---

## 1. Instalar dependências no Python 3.8

```bash
source venv/bin/activate
python -m pip install -r requirements-batch-py38.txt
```

Se o PyTorch com CUDA não instalar corretamente pelo requirements, instale separadamente:

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. Checar ambiente

Na raiz do projeto:

```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/setup/check_environment.py
```

O ideal é ver:

```text
CUDA disponível no PyTorch: OK
Subset all: encontrado=543
Subset 1st: encontrado=118
Subset 2nd: encontrado=80
Subset laines: encontrado=67
Subset arcanjo: encontrado=75
```

---

## 3. Configurar um dataset

Cada dataset tem um `.env` em:

```text
configs/datasets/<dataset>.env
```

Exemplo para KSL:

```bash
DATASET="ksl"
DATA_CSV="data/interim/ksl/ksl_mediapipe.csv"
RESULTS_DIR="experiments/ksl_nested_lopo_resume_grid"
REPORTS_DIR="reports/ksl"
SUBSETS="2nd arcanjo laines 1st all"
IMPUTATIONS="true false"
PROTOCOL="nested_lopo"
EPOCHS="30"
PATIENCE="5"
BATCH_SIZE="64"
LR="0.0001"
WD="0.0001"
DEVICE="cuda"
MAX_RUNS="none"
```

Para testar MINDS ou UFOP, edite:

```text
configs/datasets/minds.env
configs/datasets/ufop.env
```

e garanta que os CSVs existam nos caminhos configurados.

---

## 4. Validar CSV antes de treinar

```bash
python scripts/preprocessing/validate_dataset_csv.py \
  --data-csv data/interim/minds/minds_mediapipe.csv
```

O script verifica se existem colunas equivalentes a:

- `person`
- `category`
- `video_name`
- `frame`
- colunas de landmarks com sufixos `_x`, `_y`, `_z`

Caso seu CSV use nomes diferentes, configure no `.env`:

```bash
PERSON_COL="participant_id"
CATEGORY_COL="sign_id"
VIDEO_COL="sequence_id"
FRAME_COL="frame_id"
```

---

## 5. Teste pequeno de um dataset

Exemplo com KSL:

```bash
python scripts/batch/run_dataset_batch.py \
  --dataset ksl \
  --data-csv data/interim/ksl/ksl_mediapipe.csv \
  --results-dir experiments/ksl_nested_lopo_resume_grid \
  --subset 2nd \
  --imputation true \
  --max-runs 2 \
  --device cuda
```

Exemplo com MINDS:

```bash
python scripts/batch/run_dataset_batch.py \
  --dataset minds \
  --data-csv data/interim/minds/minds_mediapipe.csv \
  --results-dir experiments/minds_nested_lopo_resume_grid \
  --subset 2nd \
  --imputation true \
  --max-runs 2 \
  --device cuda
```

---

## 6. Rodar todas as condições de um dataset

```bash
./scripts/batch/run_dataset_all.sh ksl
```

Ou:

```bash
./scripts/batch/run_dataset_all.sh include50
./scripts/batch/run_dataset_all.sh minds
./scripts/batch/run_dataset_all.sh ufop
```

Esse comando usa o arquivo:

```text
configs/datasets/<dataset>.env
```

---

## 7. Rodar somente com imputação

Como no artigo os resultados principais são apresentados com imputação, você pode rodar somente essa condição:

```bash
./scripts/batch/run_dataset_imputation_only.sh minds
```

Isso usa os subsets configurados, mas força:

```text
IMPUTATIONS=true
```

---

## 8. Usar tmux

```bash
tmux new -s minds
./scripts/batch/run_dataset_all.sh minds
```

Para sair sem parar:

```text
Ctrl+B, depois d
```

Para voltar:

```bash
tmux attach -t minds
```

---

## 9. Contar resultados

```bash
python scripts/utils/count_result_files.py \
  --results-dir experiments/minds_nested_lopo_resume_grid
```

---

## 10. Sumarizar resultados

```bash
python scripts/batch/summarize_dataset_results.py \
  --dataset minds \
  --results-dir experiments/minds_nested_lopo_resume_grid \
  --reports-dir reports/minds \
  --expected-runs-per-condition 132
```

Expected runs por condição no protocolo nested LOPO:

```text
MINDS: 12 × 11 = 132
UFOP:  5 × 4  = 20
KSL:   20 × 19 = 380
```

Para INCLUDE-50, depende do número de sinalizadores no seu CSV.

---

## 11. Consolidar todos os datasets

Depois de gerar `summary_outer.csv` em cada pasta de relatório:

```bash
python scripts/analysis/summarize_all_datasets.py
```

Saída:

```text
reports/all_datasets_summary_outer.csv
```

---

## Observação importante

O script genérico assume que os datasets já foram convertidos para CSV de landmarks compatível com o pipeline. Ele **não extrai landmarks de vídeo**. Para treino batch, o MediaPipe não precisa estar instalado se o arquivo `src/preprocessing/landmark_subsets.py` já não depender de `import mediapipe`.

--

# Scripts

Estrutura pensada para rodar datasets diferentes com o mesmo batch genérico.

- `batch/`: execução e sumarização de experimentos.
- `setup/`: checagem de ambiente.
- `preprocessing/`: validação/preparação de CSVs.
- `analysis/`: consolidação/análise após os resultados.
- `utils/`: inspeções rápidas de GPU, resultados e JSONs.
