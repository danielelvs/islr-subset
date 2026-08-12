from __future__ import annotations

"""Per-video interpolation for missing landmark coordinates."""

import pandas as pd


def interpolate_series(series: pd.Series) -> pd.Series:
    """Interpolate one coordinate series.

    Cubic interpolation is used when at least four valid points are available;
    otherwise linear interpolation is used. At most five consecutive missing
    values are filled in either direction.
    """

    method = "cubic" if series.notna().sum() >= 4 else "linear"
    return series.interpolate(method=method, limit=5, limit_direction="both")


def impute_by_video(
    df: pd.DataFrame,
    landmark_columns: list[str],
    *,
    column_chunk_size: int = 96,
) -> pd.DataFrame:
    """Impute landmark coordinates independently within each video.

    Columns are processed in chunks to limit peak memory use on full MediaPipe
    Holistic CSV files. Any values that remain missing are replaced with zero.
    """

    if not landmark_columns:
        return df.copy()
    if "video_name" not in df.columns:
        raise ValueError("The DataFrame must contain a 'video_name' column.")

    result = df.copy()
    grouped = result.groupby("video_name", sort=False)

    for start in range(0, len(landmark_columns), column_chunk_size):
        chunk = landmark_columns[start : start + column_chunk_size]
        result[chunk] = grouped[chunk].transform(interpolate_series)

    result[landmark_columns] = result[landmark_columns].fillna(0)
    return result


impute_landmarks = impute_by_video
