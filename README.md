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
│   ├── run_include50_split.py
│   ├── run_include50_split_all.sh
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
python -m venv venv
source venv/bin/activate
python -m pip install pip --upgrade
# python -m pip install numpy opencv-python wheel testresources tensorflow seaborn matplotlib scikit-learn torch torchvision torchaudio xgboost tqdm scipy requests Pillow pydicom autopep8
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

Exemplo com INCLUDE-50, usando split fixo:

```bash
python scripts/batch/run_include50_split.py \
  --dataset include50 \
  --data-csv data/interim/include50/include50_mediapipe_with_split.csv \
  --results-dir experiments/include50_split_grid \
  --subset 2nd \
  --imputation true \
  --device cuda \
  --epochs 2 \
  --batch-size 64 \
  --patience 1
```

---

## 6. Rodar todas as condições de um dataset

```bash
./scripts/batch/run_dataset_all.sh ksl
```

Ou:

```bash
./scripts/batch/run_dataset_all.sh minds
./scripts/batch/run_dataset_all.sh ufop
```

Para INCLUDE-50, use o script específico com split fixo, pois esse dataset não possui `person_id`:

```bash
./scripts/batch/run_include50_split_all.sh
```

Esse comando usa:

```text
data/interim/include50/include50_mediapipe_with_split.csv
```

com a coluna:

```text
split = train / val / test
```

Os comandos genéricos usam o arquivo:

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

ou

```bash
tmux detach-client -t minds
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

Para INCLUDE-50, não use `expected-runs-per-condition` de LOPO. No protocolo com split fixo, cada condição gera 1 run, portanto são 10 runs no total: 5 subsets × 2 condições de imputação.

---

## 12. Consolidar todos os datasets

Depois de gerar `summary_outer.csv` em cada pasta de relatório:

```bash
python scripts/analysis/summarize_all_datasets.py
```

Saída:

```text
reports/all_datasets_summary_outer.csv
```

---

## 11. INCLUDE-50 com split fixo

O INCLUDE-50 usado aqui não possui `person_id`. Portanto, ele não deve ser treinado com nested LOPO.

Use primeiro o CSV com split:

```text
data/interim/include50/include50_mediapipe_with_split.csv
```

Teste uma condição:

```bash
python scripts/batch/run_include50_split.py \
  --dataset include50 \
  --data-csv data/interim/include50/include50_mediapipe_with_split.csv \
  --results-dir experiments/include50_split_grid \
  --subset 2nd \
  --imputation true \
  --device cuda \
  --epochs 2 \
  --batch-size 64 \
  --patience 1
```

Depois rode todas as condições:

```bash
./scripts/batch/run_include50_split_all.sh
```

Esse script faz a adaptação interna:

```text
category   <- sign_id
video_name <- sequence_id
frame      <- frame_id
person     <- split
```

Assim, o `Trainer` atual consegue separar treino, validação e teste sem alterar `src/training/trainer.py`.

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

## Speed Evaluation / Checkpoint Evaluation

O script `scripts/analysis/evaluate_checkpoint_speed.py` avalia um checkpoint treinado (`.pth` ou `.pt`) e mede tanto as métricas de classificação quanto o tempo de inferência.

Ele calcula:

- Accuracy
- Precision weighted
- Recall weighted
- F1-score weighted
- Tempo total de predição
- Tempo médio por amostra
- Samples per second (SPS)
- Estatísticas de tempo por batch
- Matriz de confusão opcional

Por padrão, o script processa os dados em memória para evitar criar CSVs intermediários grandes. Use `--cache-preprocessed` apenas se houver espaço em disco suficiente.

---

### 1. Pré-requisitos

Ative o ambiente virtual e configure o `PYTHONPATH`:

```bash
cd /home/danielevs/islr-subset
source venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

Confira se CUDA está disponível:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

---

### 2. Encontrar checkpoints disponíveis

O speed evaluation precisa de um modelo salvo. Para procurar checkpoints:

```bash
find experiments -name "*.pth" -o -name "*.pt"
```

Se nenhum arquivo aparecer, significa que os experimentos foram executados sem salvar o modelo. Nesse caso, é necessário treinar novamente uma condição com salvamento de checkpoint habilitado.

---

### 3. Avaliação do INCLUDE-50

O INCLUDE-50 não possui identificador de pessoa/sinalizador. Portanto, a avaliação usa a coluna `split`.

Para avaliar o conjunto de teste:

```bash
python scripts/analysis/evaluate_checkpoint_speed.py \
  --dataset include50 \
  --data-csv data/interim/include50/include50_mediapipe_with_split.csv \
  --checkpoint-path CAMINHO/DO/CHECKPOINT.pth \
  --subset 2nd \
  --imputation true \
  --category-col sign_id \
  --video-col sequence_id \
  --frame-col frame_id \
  --person-col split \
  --eval-person test \
  --device cuda \
  --batch-size 1 \
  --save-confusion-matrix
```

Nesse caso, o script interpreta:

```text
category   ← sign_id
video_name ← sequence_id
frame      ← frame_id
person     ← split
```

Assim:

```text
split == test
```

é usado como conjunto de avaliação.

---

### 4. Avaliação de datasets com LOPO: KSL, MINDS e UFOP

Para datasets com identificador de pessoa/sinalizador, o script deve avaliar a mesma pessoa usada como teste no fold do checkpoint.

Exemplo geral:

```bash
python scripts/analysis/evaluate_checkpoint_speed.py \
  --dataset ksl \
  --data-csv data/interim/ksl/ksl_mediapipe.csv \
  --checkpoint-path CAMINHO/DO/CHECKPOINT.pth \
  --subset 2nd \
  --imputation true \
  --category-col sign_id \
  --video-col sequence_id \
  --frame-col frame_id \
  --person-col person \
  --eval-person ID_DA_PESSOA_TESTE \
  --device cuda \
  --batch-size 1 \
  --save-confusion-matrix
```

Se o checkpoint foi treinado em um fold salvo como:

```text
test=3__val=7
```

então a avaliação deve usar:

```bash
--eval-person 3
```

Para MINDS ou UFOP, ajuste os nomes das colunas conforme o CSV. Por exemplo, se o CSV já usa `category`, `video_name`, `frame` e `person`:

```bash
python scripts/analysis/evaluate_checkpoint_speed.py \
  --dataset minds \
  --data-csv data/interim/minds/minds_mediapipe.csv \
  --checkpoint-path CAMINHO/DO/CHECKPOINT.pth \
  --subset 2nd \
  --imputation true \
  --category-col category \
  --video-col video_name \
  --frame-col frame \
  --person-col person \
  --eval-person ID_DA_PESSOA_TESTE \
  --device cuda \
  --batch-size 1 \
  --save-confusion-matrix
```

---

### 5. Saídas geradas

Por padrão, os resultados são salvos em:

```text
reports/speed/<dataset>_<subset>_<imputation>_<timestamp>/
```

A pasta contém:

```text
speed_eval_results.json
confusion_matrix.png
```

A matriz de confusão só é salva quando a opção abaixo é usada:

```bash
--save-confusion-matrix
```

---

### 6. Observações importantes

Use `--batch-size 1` para medir latência por amostra de forma mais direta.

Use batches maiores apenas se o objetivo for medir throughput, ou seja, quantas amostras por segundo o modelo processa em lote.

O script executa warmup antes da medição para reduzir instabilidade inicial da GPU. O padrão é:

```bash
--warmup-iters 20
```

Para evitar uso extra de disco, o script não salva o CSV pré-processado por padrão. Caso queira reutilizar o mesmo pré-processamento em várias avaliações e tenha espaço disponível, use:

```bash
--cache-preprocessed
```
