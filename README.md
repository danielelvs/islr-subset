# ISLR Landmark Subset Experiments

Official experiment repository for:

> **[Proper Body Landmark Subset Enables More Accurate and 5X Faster Recognition of Isolated Signs in LIBRAS](https://arxiv.org/abs/2510.24887)**  
> Daniele L. V. dos Santos, Thiago B. Pereira, Carlos Eduardo G. R. Alves,
> Richard J. M. G. Tello, Francisco de A. Boldt, and Thiago M. Paixão  
> arXiv:2510.24887, 2025; accepted for presentation at IEEE SAS 2026.

The paper investigates lightweight body landmark extraction for isolated sign
language recognition. Replacing OpenPose directly with MediaPipe improves
speed but can reduce accuracy. This project evaluates landmark subset selection
and spline-based missing-landmark imputation to recover recognition quality
while reducing processing time by more than 5×.

The work builds on the Skeleton-DML representation introduced by Alves, Boldt,
and Paixão in *Enhancing Brazilian Sign Language Recognition through Skeleton
Image Representation*. The current codebase uses MediaPipe, adds multiple
landmark subsets, supports four datasets, and separates the evaluation protocol
appropriate to each source.

Main contributions implemented here:

- comparison of the complete MediaPipe Holistic skeleton with four landmark subsets;
- spline-based imputation evaluated independently from subset selection;
- signer-independent evaluation for datasets with signer metadata;
- fixed-split evaluation for INCLUDE-50;
- accuracy, F1-score, extraction latency, and inference-speed analysis; and
- a harmonized publication schema for the four processed landmark datasets.

The project supports two evaluation protocols:

- **Nested LOPO** for datasets with signer identifiers: MINDS-Libras, LIBRAS-UFOP, and KSL.
- **Fixed train/validation/test split** for INCLUDE-50, because the available version does not provide a reliable signer identifier.

## Contents

- [Method and pipeline](#method-and-pipeline)
- [Landmark subsets](#landmark-subsets)
- [Datasets and protocols](#datasets-and-protocols)
- [Repository structure](#repository-structure)
- [Environment setup](#1-environment-setup)
- [Validation and extraction](#3-validate-a-landmark-csv)
- [Preprocessing](#5-filter-landmarks-and-apply-interpolation)
- [Running experiments](#7-run-the-configured-experiment-grids)
- [Reports and speed evaluation](#10-summarize-results)
- [Publication dataset](#publication-dataset)
- [Citation](#citation)
- [License](#license)

## Method and pipeline

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

Skeleton-DML transforms a temporal landmark sequence into an image consumed by
a 2D classifier. Subset selection happens before image generation. Optional
interpolation is applied independently inside each video so that information
cannot cross sample boundaries.

## Landmark subsets

| Subset    |                Landmarks | Description                                                              |
| --------- | -----------------------: | ------------------------------------------------------------------------ |
| `all`     |                      543 | Full MediaPipe Holistic output                                           |
| `1st`     |                      118 | Adapted from the first-place Google ASL Signs solution                   |
| `2nd`     |                       80 | Adapted from the second-place Google ASL Signs solution                  |
| `laines`  | 67 source / 68 effective | Selected face, upper-body pose, both hands, and a derived chest midpoint |
| `arcanjo` |                       75 | Full pose and both hands, without face landmarks                         |

The names `1st` and `2nd` refer to landmark selections adapted from the first-
and second-place Google ASL Signs solutions. They do not indicate rankings
produced by this repository.

## Datasets and protocols

The experiment runner treats every dataset as a separate benchmark. Classes
with similar names are not aligned across languages or sources.

| Dataset | Language | Frames in the current CSV | Samples/segments | Classes | Available signer IDs | Protocol |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [INCLUDE-50](https://huggingface.co/datasets/ai4bharat/INCLUDE) | Indian Sign Language | 61,613 | 929 | 50 | unavailable | official fixed split |
| [KSL](https://github.com/Yangseung/KSL) | Korean Sign Language | 108,373 | 1,229 | 67 | 20 | nested LOPO |
| [MINDS-Libras](https://zenodo.org/records/2667329) | Brazilian Sign Language | 109,392 | 800 | 20 | 8 | nested LOPO |
| [LIBRAS-UFOP](https://www.sciencedirect.com/science/article/pii/S0957417420309143) | Brazilian Sign Language | 115,656 | 3,040 | 56 | 5 | nested LOPO |

These values were calculated from the processed CSVs in `data/interim`. They
describe this project snapshot and may differ from the complete source corpora
or from the figures reported in their papers. Protocol sizes must be derived
from the signer IDs actually present in the input file.

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
├── merge_mediapipe_datasets.ipynb
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
│   ├── run_pipeline.sh
│   └── summarize_dataset_results.py
├── preprocessing/
│   ├── select_published_dataset.py
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

Keep extraction and training dependencies in separate environments. MediaPipe
is required only by the extraction environment. OpenPose is retained as legacy
reference code and is not exposed by the supported command-line pipeline.

Check the GPU directly:

```bash
python scripts/utils/check_gpu_usage.py
```

Run the core regression tests:

```bash
python -m unittest discover -s tests -v
```

## 2. Dataset protocols

The four datasets are evaluated as separate benchmarks. Their class identifiers
are local to each source and INCLUDE-50 uses a different evaluation protocol, so
`all` means "run every benchmark", not "concatenate the rows and train one model".

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
| MINDS-Libras |       8 |                 56 |
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
  --results-dir experiments/ksl_nested_lopo_grid_50 \
  --subset 2nd \
  --imputation true \
  --max-runs 1 \
  --epochs 2 \
  --patience 1 \
  --batch-size 64 \
  --device cuda \
  --person-col interpreter \
  --category-col sign_id \
  --video-col video_name \
  --frame-col frame_id
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

## 7. Run the configured experiment grids

The main entry point validates the input files and dispatches each dataset to
the correct protocol. Run a preflight before committing GPU time:

```bash
./scripts/batch/run_pipeline.sh all --dry-run
```

Run every configured subset/imputation condition for KSL only:

```bash
./scripts/batch/run_pipeline.sh ksl
```

Run all four datasets:

```bash
./scripts/batch/run_pipeline.sh all
```

This runs INCLUDE-50 with its fixed split and KSL, MINDS-Libras, and
LIBRAS-UFOP with their configured LOPO protocols. It runs them sequentially and
resumes completed folds/results. With the current nested-LOPO configuration,
the complete grid is large: KSL alone has 380 folds per condition and 10
conditions (five landmark subsets times two imputation settings).

Edit `configs/datasets/<dataset>.env` before starting to change the device,
epochs, subsets, imputation settings, cache behavior, or checkpoint saving.
The explicit `PERSON_COL`, `CATEGORY_COL`, `VIDEO_COL`, and `FRAME_COL` values in
each LOPO configuration are part of the dataset protocol and should stay aligned
with the source CSV schema.

The publication bundle produced by
`notebooks/merge_mediapipe_datasets.ipynb` is intended for distribution and
cross-dataset analysis. The experiment runner deliberately uses the original
per-dataset CSV files: `frames.csv` combines incompatible evaluation protocols,
while labels and provenance live in its auxiliary tables.

To reproduce an experiment from the publication bundle, first stream one
dataset into its own canonical view:

```bash
python scripts/preprocessing/select_published_dataset.py \
  --bundle-dir data/processed/multilingual-islr-mediapipe \
  --dataset ksl \
  --output data/processed/publication_views/ksl_frames.csv
```

That view uses `class_id`, `sample_id`, `signer_id`, and `frame_id`. These are
accepted by the central preprocessing code and can be passed explicitly through
the batch CLI column options. See `PIPELINE_AUDIT.md` for the complete data-flow
review and the publication-schema compatibility boundary.

### Lower-level commands

#### KSL, MINDS-Libras, or LIBRAS-UFOP

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

#### INCLUDE-50

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
  --results-dir experiments/ksl_nested_lopo_grid_50
```

Inspect one result:

```bash
python scripts/utils/inspect_result_json.py \
  experiments/ksl_nested_lopo_grid_50/runs/2nd/with_imputation/test=0__val=1/result.json
```

## 10. Summarize results

### Nested LOPO

```bash
python scripts/batch/summarize_dataset_results.py \
  --dataset ksl \
  --results-dir experiments/ksl_nested_lopo_grid_50 \
  --reports-dir reports/ksl_50 \
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
  --checkpoint-path experiments/ksl_nested_lopo_grid_50/runs/2nd/with_imputation/test=3__val=7/best_model.pth \
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

## Publication dataset

The landmark files can be harmonized into a publication bundle with
`notebooks/merge_mediapipe_datasets.ipynb`. It generates:

```text
data/processed/multilingual-islr-mediapipe/
├── frames.csv
├── classes.csv
├── samples.csv
├── data_dictionary.csv
├── manifest.json
└── README.md
```

The bundle normalizes identifiers and removes redundant frame-level metadata.
It is intended for distribution and cross-dataset analysis. Training still
operates on one source at a time because label spaces and evaluation protocols
are not interchangeable. The Hugging Face dataset card is available at
[`data/README.md`](data/README.md), and the schema decisions are documented in
[`notebooks/merge_mediapipe_data_dictionary.md`](notebooks/merge_mediapipe_data_dictionary.md).

## Result provenance

Only compare reports produced with the same landmark extractor, data version,
split, preprocessing, and metric aggregation. The Alves KSL split uses five
groups: 16 signers for training and the same held-out group of four signers for
both validation and test in each fold. This behavior is reproduced intentionally
by `scripts/batch/run_ksl_alves_split.py` for direct comparison with the earlier
implementation. Because validation and test coincide, report that limitation
explicitly. The main KSL configuration uses nested LOPO with distinct validation
and test signers.

Run the exact Alves KSL split with:

```bash
./scripts/batch/run_ksl_alves_split_all.sh
python scripts/batch/summarize_ksl_alves_split.py
```

## Important implementation notes

- Landmark subset selection is performed before image generation.
- The Laines subset selects 67 source landmarks. During preprocessing, a synthetic chest midpoint is computed from the two shoulder landmarks, resulting in 68 effective points for the model.
- The dataset class does not silently remove MediaPipe pose landmarks after subset selection.
- Training checkpoints are selected by the lowest validation loss, matching the early-stopping criterion.
- The requested device is respected; `--device cpu` no longer silently selects CUDA or MPS.
- Optional timm models are imported lazily, so ResNet-18 training does not fail when timm is absent.
- INCLUDE-50 never uses `sample_id` as a person identifier.

See [CODE_REVIEW.md](CODE_REVIEW.md) for the review findings and behavior-changing fixes applied to this version.

## Citation

If you use this repository, its MediaPipe landmark subsets, or its experimental
pipeline, cite the associated paper:

```bibtex
@article{dosSantos2025proper,
  title   = {Proper Body Landmark Subset Enables More Accurate and 5X Faster Recognition of Isolated Signs in LIBRAS},
  author  = {dos Santos, Daniele L. V. and Pereira, Thiago B. and Alves, Carlos Eduardo G. R. and Tello, Richard J. M. G. and Boldt, Francisco de A. and Paix{\~a}o, Thiago M.},
  journal = {arXiv preprint arXiv:2510.24887},
  year    = {2025},
  doi     = {10.48550/arXiv.2510.24887}
}
```

The paper is also accepted for presentation at IEEE SAS 2026. Update the venue
fields above when the proceedings citation becomes available.

Skeleton-DML originates from the following work, which should also be cited
when that representation is used:

```bibtex
@article{alves2024enhancing,
  title   = {Enhancing Brazilian Sign Language Recognition through Skeleton Image Representation},
  author  = {Alves, Carlos Eduardo G. R. and Boldt, Francisco de Assis and Paix{\~a}o, Thiago M.},
  journal = {arXiv preprint arXiv:2404.19148},
  year    = {2024}
}
```

Also cite the original dataset publications for every source used in an
experiment. The processed landmark bundle does not replace source attribution
or source-specific usage terms.

## License

The source code is distributed under the [MIT License](LICENSE). The
harmonized landmark release is also published under MIT by the dataset
maintainer. Cite the original datasets represented in each experiment and
retain their source attribution.
