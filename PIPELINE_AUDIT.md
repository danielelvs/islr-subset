# Pipeline and publication-schema audit

Audit scope: every Python file under `src/`, `scripts/`, and `tests/`, excluding
the virtual environment. The review followed data from raw-video discovery to
landmark extraction, schema normalization, subset selection, imputation,
dataset splitting, training, checkpoint evaluation, and report aggregation.

## Schema boundary

Extraction keeps each source dataset's native metadata. The publication
notebook is the boundary that converts those inputs to the canonical schema:

```text
dataset, class_id, sample_id, sequence_id, signer_id, frame_id,
source_frame_id, split, missing_hand_0, missing_hand_1, missing_pose,
missing_face, <1,629 coordinates>
```

Training internally normalizes either native metadata or canonical publication
metadata to four names: `category`, `video_name`, `person`, and `frame`.
Coordinate columns are then selected and renamed to the model-facing schema.

The combined publication file must not be passed unfiltered to LOPO. First
select one dataset. INCLUDE-50 uses `split` as the partition key; KSL, MINDS,
and UFOP use `signer_id`. Classes are not aligned across sources.

## Reviewed components

| Area | Files reviewed | Schema conclusion |
| --- | --- | --- |
| Dataset discovery/loaders | `src/datasets/*.py` | Native and canonical KSL/INCLUDE-50 metadata are supported. |
| Extraction | `src/extraction/*.py`, `scripts/01_extract_landmarks.py`, `scripts/05_latency_benchmark.py` | Produces native intermediate schemas; publication conversion happens later. |
| Preprocessing | `src/preprocessing/*.py`, `scripts/02_filter_landmarks.py`, `scripts/preprocessing/*.py` | Canonical aliases are supported; invalid/missing columns and duplicate frame keys now fail. |
| Representations | `src/representations/*.py` | Consumes coordinate arrays and is independent of metadata column names. |
| Models | `src/models/*.py` | Consumes images/tensors and is independent of CSV schemas. |
| Training | `src/training/*.py`, `scripts/03_train.py` | Uses normalized internal names; single-run loading now maps every source explicitly. |
| Batch protocols | `scripts/batch/*.py` | LOPO and fixed-split paths use explicit source columns. The comparison split reproduces the Alves KSL 16/4 grouping. |
| Evaluation/reporting | `scripts/analysis/*.py`, `scripts/04_show_results.py`, `scripts/utils/*.py` | Checkpoint evaluation accepts configurable columns; report aggregation does not read landmark datasets. |
| Environment/tests | `scripts/setup/*.py`, `tests/*.py` | Config paths and both native/canonical normalization are checked. |

## Material fixes from the audit

1. Added `class_id`, `sample_id`, and `signer_id` support to the central
   normalizer while preserving native-source precedence.
2. Added rejection of negative/duplicate frame keys and inconsistent
   class/signer metadata within a sample.
3. Missing landmark columns and nonnumeric coordinate text no longer become
   zeros silently. Actual missing values may still be imputed or filled by the
   selected experiment setting.
4. Fixed a `TypeError` in checkpoint evaluation caused by applying `len()` to
   an integer landmark count.
5. Preserved two distinct KSL protocols: nested LOPO uses different validation
   and test signers, while `run_ksl_alves_split.py` intentionally reproduces
   the Alves five-fold split with 16 training signers and the same four signers
   used for validation and test.
6. Made INCLUDE-50 checkpoint/training columns configurable, allowing an
   extracted canonical publication view.
7. Reworked `scripts/03_train.py` to load all four native intermediate CSVs
   through the same central preprocessing path with explicit schemas.
8. Added a streaming selector for extracting one dataset from a published
   combined bundle without loading the complete file into memory.

## Execution paths

Validate all native inputs without training:

```bash
./scripts/batch/run_pipeline.sh all --dry-run
```

Run all configured KSL conditions:

```bash
./scripts/batch/run_pipeline.sh ksl
```

Run every dataset as a separate benchmark:

```bash
./scripts/batch/run_pipeline.sh all
```

Extract KSL from a generated publication bundle:

```bash
python scripts/preprocessing/select_published_dataset.py \
  --bundle-dir data/processed/multilingual-islr-mediapipe \
  --dataset ksl \
  --output data/processed/publication_views/ksl_frames.csv
```

The extracted view uses `class_id`, `sample_id`, `signer_id`, and `frame_id`.
Pass those names explicitly to `run_dataset_batch.py`. For INCLUDE-50, use
`class_id`, `sample_id`, `split`, and `frame_id` with
`run_include50_split.py`.

## Verification

- All Python files compile.
- Shell batch scripts pass syntax validation.
- Native-input preflight passes for all four datasets.
- Tests cover native fixed-split preparation, temporal interpolation, landmark
  counts, canonical publication metadata, duplicate keys, missing schema
  columns, and the exact Alves KSL grouped split.

Full GPU training was not started during the audit. The current machine does
not expose CUDA; experiment configs remain set to CUDA for the target server.
