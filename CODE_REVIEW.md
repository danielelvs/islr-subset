# Code Review Summary

This review focused on correctness, reproducibility, runtime behavior, disk use, duplicated code, and English-language consistency in source files and command-line output.

## Critical fixes

1. **MediaPipe pose landmarks were being removed inside `LopoDataset`.**
   The previous filter used pose indices inherited from an older convention and removed shoulders, elbows, wrists, and hand-related pose points after subset selection. The implicit filter was removed. Landmark subsets are now the only source of landmark selection.

2. **The Laines strategy uses 67 source landmarks plus one derived point.**
   The subset registry now returns the original 67 source landmarks. During preprocessing, a synthetic chest midpoint is computed from the right and left shoulder landmarks, yielding 68 effective points for the model.

3. **Frame order was not preserved explicitly by `LopoDataset`.**
   The `frame` column was dropped during dataset preparation. It is now retained and each video is sorted by frame before image generation.

4. **Early stopping and checkpoint selection used different criteria.**
   Patience was based on validation loss, but weights were selected by validation accuracy. The best checkpoint is now selected by the lowest validation loss.

5. **Validation loss was averaged by batch instead of by sample.**
   The loss is now accumulated using batch size and divided by the number of validation samples.

   The dedicated Alves grouped compatibility runner explicitly enables the
   earlier last-validation-batch loss and validation-accuracy checkpoint rule
   so its historical five-fold results remain reproducible. All maintained
   LOPO and fixed-split runners use the corrected default.

6. **The command-line device option was not respected.**
   Batch scripts only printed the requested device while `Trainer` selected a device independently. The configured device is now passed to and enforced by `Trainer`.

7. **INCLUDE-50 used `sample_id` as a fake signer identifier in one loader.**
   The loader now requires `include50_mediapipe_with_split.csv` and maps `split` to the internal grouping column.

8. **Importing ResNet-18 required timm.**
   `models/__init__.py` imported every model eagerly. Model and representation modules are now loaded lazily by name.

9. **YAML values could not override command-line defaults in `03_train.py`.**
   Parser defaults were always truthy and overrode the YAML file. CLI values are now optional and take precedence only when explicitly provided.

10. **The extraction script used an inconsistent V-LIBRASIL dataset name.**
    `vlibrasil` is now used consistently, while the dataset registry still accepts `vlibras` as an alias.

11. **The single-run training command ignored subset selection for KSL and INCLUDE-50.**
    `scripts/03_train.py` loaded full CSV files directly for these datasets. It now uses the shared landmark preparation pipeline, so `--subset` and interpolation settings are applied consistently.

12. **Interpolation could follow raw CSV row order instead of temporal frame order.**
    Landmark rows are now sorted by `video_name` and `frame` before per-video interpolation, preventing incorrect values when a source CSV is not already ordered chronologically.

## Performance and maintainability improvements

- Repeated landmark column normalization code was moved to `src/preprocessing/landmark_dataframe.py`.
- Landmark columns are built in one DataFrame operation instead of thousands of repeated column insertions.
- Per-video index arrays are cached in `LopoDataset`, avoiding a full DataFrame scan for every sample.
- Each `LopoDataset` now stores only the rows needed by its train, validation, or test partition instead of copying the entire landmark DataFrame three times.
- Category-to-label mapping is cached instead of calling `list.index` for every item.
- Best-checkpoint weights are kept on CPU during training, reducing peak GPU memory use.
- Training now rejects partitions with fewer than two videos before BatchNorm fails inside the model.
- Saved trainer JSON files include the checkpoint path when checkpoint saving is enabled.
- Missing-value interpolation uses grouped transforms in manageable column chunks.
- Preprocessed CSV caches are optional for both batch training and speed evaluation; they are disabled by default to reduce disk use.
- Batch checkpoint saving is configurable and disabled by default to protect disk space.
- Dead commented implementations and obsolete compatibility comments were removed from core training code.
- macOS metadata files and Python bytecode caches were removed from the repository package.
- Core regression tests now verify subset counts and fixed-split dataset preparation.

## Reproducibility warning

The pose-landmark filter fix changes the actual model input. The Laines path now preserves the original 67-source-plus-one-derived-point behavior. Results affected by the pose filtering issue should be rerun before they are combined in one final results table.
