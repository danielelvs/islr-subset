QUANDO JA TEM A EXECUCAO, NAO DEVE FAZER O PREPROCESSING (INTERPOLATION/CRIAR CSV) [OPCIONAL?]

# ISLR Landmark Subset Experiments

This repository implements an isolated sign language recognition (ISLR) pipeline based on MediaPipe Holistic landmarks, landmark subset selection, optional spline interpolation, Skeleton-DML image encoding, and image classification models such as ResNet-18.

The project supports two evaluation protocols:

- **Nested LOPO** for datasets with signer identifiers: MINDS-Libras, LIBRAS-UFOP, and KSL.
- **Fixed train/validation/test split** for INCLUDE-50, because the available version does not provide a reliable signer identifier.

## Pipeline

```text
Raw videos or landmark CSV
        ↓
MediaPipe Holistic landmarks
        ↓
Landmark subset selection
        ↓
Optional per-video interpolation
        ↓
Skeleton-DML image representation
        ↓
Image classifier
        ↓
Accuracy, precision, recall, F1-score, and speed metrics
```

## Landmark subsets

| Subset    |                Landmarks | Description                                                              |
| --------- | -----------------------: | ------------------------------------------------------------------------ |
| `all`     |                      543 | Full MediaPipe Holistic output                                           |
| `1st`     |                      118 | Adapted from the first-place Google ASL Signs solution                   |
| `2nd`     |                       80 | Adapted from the second-place Google ASL Signs solution                  |
| `laines`  | 67 source / 68 effective | Selected face, upper-body pose, both hands, and a derived chest midpoint |
| `arcanjo` |                       75 | Full pose and both hands, without face landmarks                         |

## Repository structure

```text
configs/
└── datasets/
    ├── include50.env
    ├── ksl.env
    ├── minds.env
    └── ufop.env

data/
├── raw/
├── interim/
└── processed/

notebooks/
├── include50_mediapipe_extraction.ipynb
├── ksl_mediapipe_extraction.ipynb
├── pipeline.ipynb
└── ufop_mediapipe_extraction.ipynb

scripts/
├── 01_extract_landmarks.py
├── 02_filter_landmarks.py
├── 03_train.py
├── 04_show_results.py
├── 05_latency_benchmark.py
├── analysis/
│   ├── evaluate_checkpoint_speed.py
│   └── summarize_all_datasets.py
├── batch/
│   ├── run_dataset_batch.py
│   ├── run_dataset_all.sh
│   ├── run_dataset_imputation_only.sh
│   ├── run_include50_split.py
│   ├── run_include50_split_all.sh
│   └── summarize_dataset_results.py
├── preprocessing/
│   └── validate_dataset_csv.py
├── setup/
│   └── check_environment.py
└── utils/
    ├── check_gpu_usage.py
    ├── count_result_files.py
    └── inspect_result_json.py

src/
├── datasets/
├── extraction/
├── models/
├── preprocessing/
├── representations/
└── training/

tests/
└── test_core.py
```

## 1. Environment setup

Python 3.10 is recommended. The pinned PyTorch version used by this project should not be installed in a Python 3.12 virtual environment.

```bash
cd /path/to/islr-subset

/usr/bin/python3.10 -m venv venv
source venv/bin/activate

python --version
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch with CUDA support. The following command uses the CUDA 11.8 wheel index:

```bash
python -m pip install --no-cache-dir \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining training dependencies:

```bash
python -m pip install --no-cache-dir -r requirements-training-py310.txt
```

For MediaPipe video extraction, install the extraction dependencies instead of `opencv-python-headless`:

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install --no-cache-dir -r requirements-extraction-py310.txt
```

Set the source path for the current shell:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

### Verify the environment

```bash
python scripts/setup/check_environment.py
```

The subset checks should report:

```text
all:     543
1st:     118
2nd:      80
laines:   67
arcanjo:  75
```

Check the GPU directly:

```bash
python scripts/utils/check_gpu_usage.py
```

Run the core regression tests:

```bash
python -m unittest discover -s tests -v
```

## 2. Dataset protocols

### MINDS-Libras, LIBRAS-UFOP, and KSL

These datasets use signer-independent evaluation with nested leave-one-person-out validation:

```text
one signer for testing
one different signer for validation
all remaining signers for training
```

For `n` signers, each subset/imputation condition contains `n × (n - 1)` runs.

| Dataset      | Signers | Runs per condition |
| ------------ | ------: | -----------------: |
| MINDS-Libras |      12 |                132 |
| LIBRAS-UFOP  |       5 |                 20 |
| KSL          |      20 |                380 |

### INCLUDE-50

INCLUDE-50 uses the fixed split stored in the CSV column `split`:

```text
train → training
val   → validation
test  → testing
```

All frames from the same `sequence_id` must remain in the same split. Do not use `sample_id` as a signer proxy.

Expected input file:

```text
data/interim/include50/include50_mediapipe_with_split.csv
```

## 3. Validate a landmark CSV

```bash
python scripts/preprocessing/validate_dataset_csv.py \
  --data-csv data/interim/ksl/ksl_mediapipe.csv
```

For INCLUDE-50:

```bash
python scripts/preprocessing/validate_dataset_csv.py \
  --data-csv data/interim/include50/include50_mediapipe_with_split.csv
```

The validator looks for metadata aliases and coordinate columns ending in `_x`, `_y`, and `_z`.

## 4. Landmark extraction

Example for MINDS-Libras with MediaPipe:

```bash
python scripts/01_extract_landmarks.py \
  --dataset minds \
  --extractor mediapipe \
  --input-dir data/raw \
  --output-dir data/interim
```

Example for LIBRAS-UFOP:

```bash
python scripts/01_extract_landmarks.py \
  --dataset ufop \
  --extractor mediapipe \
  --input-dir data/raw \
  --output-dir data/interim
```

Dataset-specific extraction notebooks are available under `notebooks/` for KSL, INCLUDE-50, and LIBRAS-UFOP.

## 5. Filter landmarks and apply interpolation

```bash
python scripts/02_filter_landmarks.py \
  --datasets minds ufop \
  --subsets all 1st 2nd laines arcanjo
```

Run an ablation without interpolation:

```bash
python scripts/02_filter_landmarks.py \
  --datasets minds \
  --subsets 2nd \
  --no-imputation
```

Interpolation is performed independently within each video. Cubic interpolation is used when at least four valid points are available; otherwise linear interpolation is used. The maximum interpolation gap is five consecutive frames in either direction.

## 6. Small training tests

Always run a short test before launching the full grid.

### Nested LOPO example

```bash
python scripts/batch/run_dataset_batch.py \
  --dataset ksl \
  --data-csv data/interim/ksl/ksl_mediapipe.csv \
  --results-dir experiments/ksl_nested_lopo_resume_grid \
  --subset 2nd \
  --imputation true \
  --max-runs 1 \
  --epochs 2 \
  --patience 1 \
  --batch-size 64 \
  --device cuda
```

### INCLUDE-50 fixed split example

```bash
python scripts/batch/run_include50_split.py \
  --data-csv data/interim/include50/include50_mediapipe_with_split.csv \
  --results-dir experiments/include50_split_grid \
  --subset 2nd \
  --imputation true \
  --epochs 2 \
  --patience 1 \
  --batch-size 64 \
  --device cuda
```

## 7. Run all conditions

### KSL, MINDS-Libras, or LIBRAS-UFOP

Edit the corresponding file under `configs/datasets/`, then run:

```bash
./scripts/batch/run_dataset_all.sh ksl
./scripts/batch/run_dataset_all.sh minds
./scripts/batch/run_dataset_all.sh ufop
```

Run only the interpolation-enabled conditions:

```bash
./scripts/batch/run_dataset_imputation_only.sh ksl
```

### INCLUDE-50

```bash
chmod +x scripts/batch/run_include50_split_all.sh
./scripts/batch/run_include50_split_all.sh
```

The INCLUDE-50 grid contains ten experiments:

```text
5 subsets × 2 interpolation conditions = 10 runs
```

By default, batch scripts do not save model checkpoints because thousands of LOPO checkpoints can consume substantial disk space. They also keep preprocessed landmark data in memory instead of writing one large cache CSV per subset/imputation condition.

To save checkpoints, set this in the dataset configuration:

```bash
SAVE_MODELS="true"
```

For INCLUDE-50, saving all checkpoints means one checkpoint per condition. For nested LOPO datasets, it means one checkpoint per fold.

To persist preprocessed CSV files for faster restarts, enable this only when sufficient disk space is available:

```bash
CACHE_PREPROCESSED="true"
```

The default is `false`. Existing caches under `experiments/<dataset>/preprocessed/` can be removed without deleting trained results.

## 8. Run experiments in tmux

```bash
tmux new -s include50
```

Inside tmux:

```bash
cd /path/to/islr-subset
source venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
./scripts/batch/run_include50_split_all.sh
```

Detach without stopping the process:

```text
Ctrl+B, then D
```

Reattach:

```bash
tmux attach -t include50
```

## 9. Inspect progress

Count result files:

```bash
python scripts/utils/count_result_files.py \
  --results-dir experiments/ksl_nested_lopo_resume_grid
```

Inspect one result:

```bash
python scripts/utils/inspect_result_json.py \
  experiments/ksl_nested_lopo_resume_grid/runs/2nd/with_imputation/test=0__val=1/result.json
```

## 10. Summarize results

### Nested LOPO

```bash
python scripts/batch/summarize_dataset_results.py \
  --dataset ksl \
  --results-dir experiments/ksl_nested_lopo_resume_grid \
  --reports-dir reports/ksl \
  --expected-runs-per-condition 380
```

### INCLUDE-50 fixed split

```bash
python scripts/batch/summarize_dataset_results.py \
  --dataset include50 \
  --results-dir experiments/include50_split_grid \
  --reports-dir reports/include50 \
  --expected-runs-per-condition 1
```

Combine all dataset summaries:

```bash
python scripts/analysis/summarize_all_datasets.py
```

## 11. Checkpoint speed evaluation

The speed evaluation script reports both:

- **End-to-end timing**, including dataset loading for each sample, Skeleton-DML generation, tensor transfer, and model inference.
- **Model forward timing**, measured around the model call with CUDA synchronization.

Use a batch size of 1 for latency measurements. Use a larger batch size only when measuring throughput.

### Find available checkpoints

```bash
find experiments -type f \( -name "*.pth" -o -name "*.pt" \)
```

If no checkpoint is found, rerun the selected condition with:

```bash
--save-model true
```

### INCLUDE-50

```bash
python scripts/analysis/evaluate_checkpoint_speed.py \
  --dataset include50 \
  --data-csv data/interim/include50/include50_mediapipe_with_split.csv \
  --checkpoint-path experiments/include50_split_grid/runs/2nd/with_imputation/fixed_split/best_model.pth \
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

### LOPO dataset

The evaluation person must match the test signer used to train the checkpoint.

```bash
python scripts/analysis/evaluate_checkpoint_speed.py \
  --dataset ksl \
  --data-csv data/interim/ksl/ksl_mediapipe.csv \
  --checkpoint-path experiments/ksl_nested_lopo_resume_grid/runs/2nd/with_imputation/test=3__val=7/best_model.pth \
  --subset 2nd \
  --imputation true \
  --category-col sign_id \
  --video-col sequence_id \
  --frame-col frame_id \
  --person-col interpreter \
  --eval-person 3 \
  --device cuda \
  --batch-size 1 \
  --save-confusion-matrix
```

Outputs are stored under:

```text
reports/speed/<dataset>_<subset>_<imputation>_<timestamp>/
```

The folder may contain:

```text
speed_eval_results.json
confusion_matrix.png
```

Use `--cache-preprocessed` only when repeated evaluations justify an additional large CSV and sufficient disk space is available.

## 12. Extraction latency benchmark

```bash
python scripts/05_latency_benchmark.py \
  --dataset minds \
  --extractor mediapipe \
  --sample-size 5 \
  --input-dir data/raw
```

The benchmark selects videos across the duration distribution and stores a JSON report under `reports/tables/`.

## Important implementation notes

- Landmark subset selection is performed before image generation.
- The Laines subset selects 67 source landmarks. During preprocessing, a synthetic chest midpoint is computed from the two shoulder landmarks, resulting in 68 effective points for the model.
- The dataset class does not silently remove MediaPipe pose landmarks after subset selection.
- Training checkpoints are selected by the lowest validation loss, matching the early-stopping criterion.
- The requested device is respected; `--device cpu` no longer silently selects CUDA or MPS.
- Optional timm models are imported lazily, so ResNet-18 training does not fail when timm is absent.
- INCLUDE-50 never uses `sample_id` as a person identifier.

See [CODE_REVIEW.md](CODE_REVIEW.md) for the review findings and behavior-changing fixes applied to this version.
