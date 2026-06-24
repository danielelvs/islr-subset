from __future__ import annotations
"""
Spline / linear imputation of missing landmark values (NaN).

Separated from filtering so you can run with or without imputation
(useful for ablation experiments).
"""

import pandas as pd
from tqdm import tqdm


def interpolate_series(series: pd.Series) -> pd.Series:
    """
    Interpolate a single landmark coordinate series grouped by video.
    Uses cubic spline when ≥4 non-NaN points are available, linear otherwise.
    Limit: 5 consecutive missing frames in both directions.
    """
    if series.notna().sum() < 4:
        return series.interpolate(method="linear", limit=5, limit_direction="both")
    return series.interpolate(method="cubic", limit=5, limit_direction="both")


def impute_by_video(df: pd.DataFrame, landmark_columns: list[str]) -> pd.DataFrame:
    """
    Apply per-video imputation to all landmark columns.
    Remaining NaN after interpolation are filled with 0.
    """
    df = df.copy()
    videos = df["video_name"].unique()
    for video in tqdm(videos, desc="Imputando"):
        mask = df["video_name"] == video
        df.loc[mask, landmark_columns] = (
            df.loc[mask, landmark_columns].apply(interpolate_series)
        )
    df[landmark_columns] = df[landmark_columns].fillna(0)
    return df

    # df = df.copy()
    # df[landmark_columns] = df.groupby("video_name")[landmark_columns].transform(
    #     interpolate_series
    # )
    # df[landmark_columns] = df[landmark_columns].fillna(0)
    # return df

impute_landmarks = impute_by_video
