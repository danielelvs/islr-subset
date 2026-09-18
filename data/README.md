---
pretty_name: Multilingual ISLR MediaPipe Landmarks
language: [bzs, ins, kvk]
license: mit
task_categories: [video-classification]
tags:
  [
    sign-language-recognition,
    isolated-sign-language-recognition,
    mediapipe,
    landmarks,
    keypoints,
    multilingual,
  ]
size_categories: [100K<n<1M]
configs:
  - config_name: frames
    data_files: [{ split: train, path: frames.csv }]
  - config_name: classes
    data_files: [{ split: train, path: classes.csv }]
  - config_name: samples
    data_files: [{ split: train, path: samples.csv }]
---

# Multilingual ISLR MediaPipe Landmarks

## Dataset Description

This dataset combines frame-level MediaPipe Holistic landmarks derived from four isolated sign language recognition (ISLR) resources: **INCLUDE-50**, **KSL**, **MINDS-Libras**, and **LIBRAS-UFOP**. It provides a common tabular schema for research on landmark selection, temporal modeling, signer-independent evaluation, and multilingual transfer learning.

The release contains landmarks rather than source RGB videos. Every frame has 543 landmarks with three coordinates each: 468 face landmarks, 21 landmarks for each hand, and 33 pose landmarks. Labels, sample provenance, and repeated frame metadata are normalized into separate tables.

This is a harmonized collection of derived artifacts. It does not assert that the source corpora share recording conditions, coordinate reference systems, class semantics, participant populations, or licenses.

- **Languages:** Brazilian Sign Language (Libras), Indian Sign Language (ISL), and Korean Sign Language (KSL)
- **Task:** isolated sign language recognition from landmark sequences
- **Unit of observation:** one video frame
- **Landmark extractor:** MediaPipe Holistic
- **Coordinates per frame:** 1,629 (`543 landmarks × 3 axes`)
- **Total columns in `frames.csv`:** 1,641
- **Processing repository:** [danielelvs/islr-subset](https://github.com/danielelvs/islr-subset)

## Release Statistics

These counts were calculated directly from the four processed CSVs used to build this release. They describe the available derived data and can differ from counts reported for the original corpora.

| Source       | Sign language           |      Frames |        Samples | Sequences |                      Classes |              Signer IDs | Evaluation metadata     |
| ------------ | ----------------------- | ----------: | -------------: | --------: | ---------------------------: | ----------------------: | ----------------------- |
| INCLUDE-50   | Indian Sign Language    |      61,613 |            929 |       929 |                           50 |             unavailable | fixed split             |
| KSL          | Korean Sign Language    |     108,373 |          1,229 |     1,229 |                           67 |                      20 | signer ID               |
| MINDS-Libras | Brazilian Sign Language |     109,392 |            800 |       800 |                           20 |                       8 | signer ID               |
| LIBRAS-UFOP  | Brazilian Sign Language |     115,656 | 3,040 segments |       280 |                           56 |                       5 | signer and sequence IDs |
| **Total**    | —                       | **395,034** |      **5,998** | **3,238** | **193 source-local classes** | **33 source-local IDs** | —                       |

Classes and signer IDs are source-local. Semantically similar labels are not merged, translated, or treated as identical across datasets.

### INCLUDE-50 split

Only INCLUDE-50 supplies a release split. It is preserved without modification, with all frames from a sequence kept together.

| Split              | Frames | Samples |
| ------------------ | -----: | ------: |
| train              | 43,082 |     649 |
| validation (`val`) |  9,164 |     140 |
| test               |  9,367 |     140 |

KSL, MINDS-Libras, and LIBRAS-UFOP rows have an empty `split` field. Construct their evaluation partitions from signer and sequence groups; do not randomly split frames.

## Dataset Structure

```text
frames.csv             # one row per frame
classes.csv            # class labels and provenance
samples.csv            # video/segment provenance
data_dictionary.csv    # definition of every published column
manifest.json          # inputs, counts, and export status
README.md
```

The frame key is (`sample_id`, `frame_id`). Read identifiers as strings to preserve leading zeros. Join `frames.csv` to the lookup tables using `class_id` and `sample_id`, never labels, filenames, or source-local IDs.

### `frames.csv`

The first 12 columns are canonical metadata:

| Column            | Type    | Nullable | Description                                                                  |
| ----------------- | ------- | -------- | ---------------------------------------------------------------------------- |
| `dataset`         | string  | no       | `include50`, `ksl`, `minds`, or `ufop`.                                      |
| `class_id`        | string  | no       | Global class key; references `classes.csv`.                                  |
| `sample_id`       | string  | no       | Global video/segment key; references `samples.csv`.                          |
| `sequence_id`     | string  | no       | Groups segments from one original sequence; equals `sample_id` outside UFOP. |
| `signer_id`       | string  | yes      | Source-prefixed signer ID; unavailable for INCLUDE-50.                       |
| `frame_id`        | integer | no       | Non-negative frame index within the sample; gaps are preserved.              |
| `source_frame_id` | integer | yes      | Original UFOP frame index; empty elsewhere.                                  |
| `split`           | string  | yes      | `train`, `val`, or `test` for INCLUDE-50; empty elsewhere.                   |
| `missing_hand_0`  | boolean | no       | `True` if any coordinate from hand 0 is missing.                             |
| `missing_hand_1`  | boolean | no       | `True` if any coordinate from hand 1 is missing.                             |
| `missing_pose`    | boolean | no       | `True` if any pose coordinate is missing.                                    |
| `missing_face`    | boolean | no       | `True` if any face coordinate is missing.                                    |

The remaining 1,629 columns follow this order:

```text
face_0_x ... face_467_z
hand_0_0_x ... hand_0_20_z
pose_0_x ... pose_32_z
hand_1_0_x ... hand_1_20_z
```

Every landmark has `_x`, `_y`, and `_z` coordinates. Empty fields represent missing values. Zero is a valid coordinate.

### `classes.csv`

| Column                        | Description                                  |
| ----------------------------- | -------------------------------------------- |
| `class_id`                    | Global key in the form `dataset::source_id`. |
| `source_class_id`             | Original `sign_id`, or `category` for MINDS. |
| `label`                       | Original textual label when available.       |
| `category_name`               | Original INCLUDE-50 thematic category.       |
| `source_category_id`          | Original UFOP category code.                 |
| `source_class_key`            | Original INCLUDE-50/UFOP sign key.           |
| `source_class_id_in_category` | Original UFOP within-category sign ID.       |

### `samples.csv`

| Column                 | Description                                  |
| ---------------------- | -------------------------------------------- |
| `sample_id`            | Global video or segment key.                 |
| `source_sample_id`     | Original sample ID when supplied.            |
| `source_sequence_id`   | Original sequence ID when supplied.          |
| `source_signer_id`     | Original signer ID for KSL, MINDS, and UFOP. |
| `source_video_name`    | Original video name.                         |
| `source_video_relpath` | Original relative path for INCLUDE-50.       |
| `source_start_frame`   | Original inclusive UFOP segment start.       |
| `source_end_frame`     | Original inclusive UFOP segment end.         |

Global IDs contain a dataset prefix and percent-escaped source components separated by `::`.

## Data Processing

The publication pipeline harmonizes schemas without rounding, interpolation, normalization, or rescaling of finite coordinates. It:

1. validates each source header;
2. maps anatomical MINDS landmark names to MediaPipe indices;
3. creates global class, sample, sequence, and signer IDs;
4. moves repeated class and sample data into lookup tables;
5. converts empty and case-insensitive `NaN` fields to empty CSV values;
6. recomputes missing-landmark flags from the published coordinates; and
7. validates numeric values, keys, metadata consistency, and UFOP segment limits.

The ambiguous source field `missing_hand` is excluded. Derive `missing_hand_0 OR missing_hand_1` or `missing_hand_0 AND missing_hand_1` explicitly, depending on the required meaning.

Rebuild the release with `notebooks/merge_mediapipe_datasets.ipynb`. The generated `manifest.json` is authoritative for the exact inputs, counts, and completion status of a build.

## Loading the Dataset

Each table is a Hugging Face configuration:

```python
from datasets import load_dataset

repo_id = "danielelvs/multilingual-islr-mediapipe"
frames = load_dataset(repo_id, "frames", split="train")
classes = load_dataset(repo_id, "classes", split="train")
samples = load_dataset(repo_id, "samples", split="train")
```

Stream the large frame table when memory is limited:

```python
frames = load_dataset(repo_id, "frames", split="train", streaming=True)
for row in frames.take(3):
    print(row["dataset"], row["sample_id"], row["frame_id"])
```

With pandas, read selected columns in chunks:

```python
import pandas as pd

metadata = [
    "dataset", "class_id", "sample_id", "sequence_id", "signer_id",
    "frame_id", "source_frame_id", "split",
    "missing_hand_0", "missing_hand_1", "missing_pose", "missing_face",
]
for chunk in pd.read_csv("frames.csv", usecols=metadata, chunksize=50_000):
    print(chunk.groupby("dataset").size())
```

For relational joins:

```python
frames = pd.read_csv("frames.csv", dtype={"class_id": "string", "sample_id": "string"})
classes = pd.read_csv("classes.csv", dtype="string")
samples = pd.read_csv("samples.csv", dtype="string")

data = (
    frames
    .merge(classes, on="class_id", how="left", validate="many_to_one")
    .merge(samples, on="sample_id", how="left", validate="many_to_one")
)
```

## Recommended Evaluation

Frames from the same sample are temporally related and must stay in one partition. UFOP segments sharing a `sequence_id` must also remain together.

- Use the provided split for INCLUDE-50.
- Use signer-independent grouped evaluation for KSL, MINDS-Libras, and LIBRAS-UFOP, such as leave-one-person-out validation.
- Create splits independently per source unless the research question defines a cross-dataset protocol.
- Report results per source because pooled scores can conceal differences in language, label space, participants, and capture conditions.

This release does not provide a verified multilingual mapping between signs.

## Limitations, Biases, and Privacy

- These are processed subsets available to this project and may omit items from the original corpora.
- MediaPipe missingness can correlate with motion, occlusion, camera angle, skin appearance, clothing, background, and capture quality, creating systematic bias.
- Source acquisition and extraction settings can differ. Identical column names do not establish measurement equivalence.
- Coordinates are not calibrated 3D world positions unless the relevant source processing record explicitly establishes that property.
- Labels are source-provided and have not been linguistically normalized, translated, or validated across languages.
- Participant coverage is limited; results may not generalize to unseen signers, dialects, signing styles, cameras, or environments.
- INCLUDE-50 has no reliable signer ID in the available table, so signer-independent evaluation cannot be reconstructed here.
- Face landmarks and body motion can retain biometric and behavioral information. Removing RGB pixels does not make the data anonymous.

Suitable research uses include isolated sign recognition, temporal modeling, landmark subset selection, missing-data robustness, and domain-shift analysis. Do not use the release for biometric identification, signer re-identification, surveillance, high-stakes decisions about people, or claims of linguistic equivalence across sign languages. An isolated-sign classifier is not a complete sign language translation system.

## License and Source Terms

This harmonized landmark release is distributed under the MIT License by its maintainer. The release contains derived landmark coordinates rather than source RGB videos. Users must retain the source citations and comply with any attribution or use conditions that continue to apply to the original corpora.

Known source pages:

- [INCLUDE repository](https://huggingface.co/datasets/ai4bharat/INCLUDE)
- [KSL repository](https://github.com/Yangseung/KSL)
- [MINDS-Libras catalogue entry](https://live.european-language-grid.eu/catalogue/lcr/21907)
- [LIBRAS-UFOP thesis record](https://educapes.capes.gov.br/handle/capes/650935?mode=full)

<!-- ## Citation

Cite this release or its associated repository version and every original source used in an experiment. INCLUDE should at minimum be cited as:

```bibtex
@inproceedings{sridhar2020include,
  title     = {INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
  author    = {Sridhar, Advaith and Ganesan, Rohith and Kumar, Pratyush and Khapra, Mitesh M.},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
  year      = {2020},
  doi       = {10.1145/3394171.3413528}
}
```

Add the verified KSL, MINDS-Libras, and LIBRAS-UFOP citations from the exact downloaded versions before public release. Do not substitute a citation from a similarly named corpus. -->

## Reproducibility and Versioning

Each publication release should preserve the generated `manifest.json`, processing commit hash, MediaPipe version and settings per source, immutable input identifiers or checksums, exact source licenses and citations, and any filtering performed before extraction.

Counts in this card correspond to the current processed inputs. Regenerate the tables and statistics whenever inputs or harmonization code change.
