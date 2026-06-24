from __future__ import annotations
"""
Filter pipeline: select landmark subset + optional imputation → write CSV.
"""

import os
import pandas as pd

from preprocessing.landmark_subsets import SUBSETS, indices_to_columns
from preprocessing.imputation import impute_by_video

METADATA_COLUMNS = ["category", "video_name", "frame", "person", "missing_hand", "missing_face"]


def filter_and_save(
    input_path: str,
    output_path: str,
    subset_name: str,
    impute: bool = True,
    dataset_name: str = "",
) -> None:
    """
    Load a raw MediaPipe CSV, select the desired subset, optionally impute,
    and write the result to output_path.

    Args:
        input_path:   Path to <dataset>_mediapipe.csv
        output_path:  Where to save the filtered CSV
        subset_name:  One of SUBSETS keys ("all", "1st", "2nd", "laines", "arcanjo")
        impute:       Whether to run spline imputation (default: True)
        dataset_name: Used for special handling (e.g. "laines" midpoint column)
    """
    if subset_name not in SUBSETS:
        raise ValueError(f"Unknown subset '{subset_name}'. Available: {list(SUBSETS.keys())}")

    indices = SUBSETS[subset_name]()
    landmark_cols = indices_to_columns(indices)

    df = pd.read_csv(input_path, na_values=["NaN"])

    # Keep only columns that exist (graceful on partial datasets)
    meta_in_df = [c for c in METADATA_COLUMNS if c in df.columns]
    lm_in_df = [c for c in landmark_cols if c in df.columns]

    if impute:
        df = impute_by_video(df, lm_in_df)

    result = df[meta_in_df + lm_in_df].copy()

    # Special case: laines needs a midpoint column between shoulders
    if subset_name == "laines":
        from preprocessing.landmark_subsets import indices_to_columns as i2c
        right_cols = i2c([500])
        left_cols  = i2c([501])
        for r, l in zip(right_cols, left_cols):
            if r in result.columns and l in result.columns:
                axis = r[-1]  # 'x', 'y', or 'z'
                result[f"pose_middle_chest_{axis}"] = (result[r] + result[l]) / 2

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows → {output_path}")

filter_landmarks = filter_and_save
