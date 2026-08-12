from __future__ import annotations

"""Landmark subset filtering with optional per-video interpolation."""

from pathlib import Path

from preprocessing.landmark_dataframe import load_and_prepare_csv


def filter_and_save(
    input_path: str,
    output_path: str,
    subset_name: str,
    impute: bool = True,
    dataset_name: str = "",
) -> None:
    """Select a landmark subset, optionally interpolate missing values, and save it.

    ``dataset_name`` is retained for compatibility with the command-line
    pipeline. Metadata aliases are detected by the shared DataFrame
    preparation module.
    """

    del dataset_name

    source_path = Path(input_path)
    destination_path = Path(output_path)
    result = load_and_prepare_csv(
        source_path,
        subset=subset_name,
        use_imputation=impute,
    )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination_path, index=False)
    print(f"Saved {len(result):,} rows to {destination_path}")


filter_landmarks = filter_and_save
