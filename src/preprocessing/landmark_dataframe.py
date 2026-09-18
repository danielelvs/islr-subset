from __future__ import annotations

"""Utilities for normalizing and selecting MediaPipe landmark CSV columns."""

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from preprocessing.imputation import impute_by_video
from preprocessing.landmark_subsets import SUBSETS, indices_to_columns

DEFAULT_METADATA_COLUMNS = [
    "category",
    "video_name",
    "frame",
    "person",
    "missing_hand",
    "missing_face",
    "missing_hand_0",
    "missing_hand_1",
    "missing_pose",
    "sign",
    "sign_id",
    "sign_key",
    "split",
    "interpreter",
    "signer",
    "dataset",
    "class_id",
    "sample_id",
    "sequence_id",
    "signer_id",
    "source_frame_id",
]

_COLUMN_ALIASES = {
    "person": ["person", "interpreter", "signer", "participant_id", "signer_id"],
    "category": ["category", "sign_id", "label", "class", "class_id"],
    "video_name": ["video_name", "sequence_id", "path", "file", "filename", "sample_id"],
    "frame": ["frame", "frame_id"],
}

def effective_subset_landmark_count(subset_name: str) -> int:
    count = len(subset_indices(subset_name))
    if subset_name == "laines":
        count += 1
    return count


def imputation_label(use_imputation: bool) -> str:
    return "with_imputation" if use_imputation else "without_imputation"


def subset_indices(subset_name: str) -> list[int]:
    if subset_name not in SUBSETS:
        raise ValueError(
            f"Unknown subset '{subset_name}'. Available subsets: {list(SUBSETS)}"
        )
    return SUBSETS[subset_name]()


def _strip_axis(column_name: str) -> str:
    return re.sub(r"_[xyz]$", "", column_name)


def _source_base_name(idx: int) -> str:
    columns = indices_to_columns([idx])
    if not columns:
        raise ValueError(f"Invalid landmark index or missing column mapping: {idx}")
    return _strip_axis(columns[0])


def _numeric_aliases(idx: int) -> list[str]:
    if idx < 468:
        return [f"face_{idx}"]
    if 468 <= idx < 489:
        hand_idx = idx - 468
        return [
            f"hand_0_{hand_idx}",
            f"hand_0_{hand_idx:02d}",
            f"left_hand_{hand_idx}",
            f"left_hand_{hand_idx:02d}",
        ]
    if 489 <= idx < 522:
        pose_idx = idx - 489
        return [
            f"pose_{pose_idx}",
            f"pose_{pose_idx:02d}",
            f"body_{pose_idx}",
            f"body_{pose_idx:02d}",
        ]
    if 522 <= idx < 543:
        hand_idx = idx - 522
        return [
            f"hand_1_{hand_idx}",
            f"hand_1_{hand_idx:02d}",
            f"right_hand_{hand_idx}",
            f"right_hand_{hand_idx:02d}",
        ]
    raise ValueError(f"Landmark index is outside the MediaPipe Holistic range: {idx}")


def _expected_base_names(idx: int) -> list[str]:
    source_base = _source_base_name(idx)
    names = [source_base]

    if source_base.startswith("hand_0_"):
        names.append(source_base.replace("hand_0_", "left_hand_", 1))
    if source_base.startswith("hand_1_"):
        names.append(source_base.replace("hand_1_", "right_hand_", 1))
    if source_base.startswith("pose_"):
        names.append(source_base.replace("pose_", "body_", 1))

    names.extend(_numeric_aliases(idx))
    return list(dict.fromkeys(names))


def _find_coordinate_column(
    columns: Iterable[str], base_names: list[str], axis: str
) -> str | None:
    available = set(columns)
    for base_name in base_names:
        for candidate in (
            f"{base_name}_{axis}",
            f"{base_name}.{axis}",
            f"{base_name}:{axis}",
        ):
            if candidate in available:
                return candidate
    return None


def _canonical_base_name(idx: int) -> str:
    if idx < 468:
        return f"face_{idx}"
    if 468 <= idx < 489:
        return f"hand_0_{idx - 468:02d}"
    if 489 <= idx < 522:
        return f"pose_{idx - 489:02d}"
    if 522 <= idx < 543:
        return f"hand_1_{idx - 522:02d}"
    raise ValueError(f"Landmark index is outside the MediaPipe Holistic range: {idx}")


def _resolve_source_column(
    df: pd.DataFrame, canonical_name: str, explicit_name: str | None
) -> str | None:
    if explicit_name:
        if explicit_name not in df.columns:
            raise ValueError(
                f"Configured column '{explicit_name}' was not found in the CSV."
            )
        return explicit_name

    for candidate in _COLUMN_ALIASES[canonical_name]:
        if candidate in df.columns:
            return candidate
    return None


def normalize_metadata(
    raw_df: pd.DataFrame,
    *,
    person_col: str | None = None,
    category_col: str | None = None,
    video_col: str | None = None,
    frame_col: str | None = None,
    lowercase_person: bool = False,
    allowed_person_values: set[str] | None = None,
) -> pd.DataFrame:
    """Create the metadata columns required by :class:`LopoDataset`.

    The resulting DataFrame always contains ``category``, ``video_name``,
    ``frame``, and ``person``. Class labels are mapped to consecutive integers
    while the original metadata columns are retained when possible.
    """

    df = raw_df.copy()
    publication_schema = {
        "dataset", "class_id", "sample_id", "signer_id", "frame_id"
    }.issubset(df.columns)
    if publication_schema:
        person_col = person_col or "signer_id"
        category_col = category_col or "class_id"
        video_col = video_col or "sample_id"
        frame_col = frame_col or "frame_id"
    source_columns = {
        "person": _resolve_source_column(df, "person", person_col),
        "category": _resolve_source_column(df, "category", category_col),
        "video_name": _resolve_source_column(df, "video_name", video_col),
        "frame": _resolve_source_column(df, "frame", frame_col),
    }

    missing = [
        name
        for name in ("person", "category", "video_name")
        if source_columns[name] is None
    ]
    if missing:
        raise ValueError(
            f"CSV is missing required metadata columns: {missing}. "
            f"Available columns: {list(df.columns[:60])}..."
        )

    original_categories = df[source_columns["category"]]

    if pd.api.types.is_numeric_dtype(original_categories):
        category_values = sorted(
            original_categories.dropna().unique()
        )
    else:
        category_values = sorted(
            original_categories.dropna().unique(),
            key=lambda value: str(value),
        )

    category_map = {value: index for index, value in enumerate(category_values)}
    df["category"] = original_categories.map(category_map).astype(int)
    df["video_name"] = df[source_columns["video_name"]].astype(str)

    if source_columns["frame"] is None:
        df["frame"] = df.groupby("video_name", sort=False).cumcount()
    else:
        df["frame"] = pd.to_numeric(
            df[source_columns["frame"]], errors="raise"
        ).astype(int)

    person = df[source_columns["person"]].astype(str).str.strip()
    if lowercase_person:
        person = person.str.lower()
    df["person"] = person

    if (df["frame"] < 0).any():
        raise ValueError("Frame identifiers must be non-negative integers.")
    duplicate_frames = df.duplicated(["video_name", "frame"], keep=False)
    if duplicate_frames.any():
        examples = (
            df.loc[duplicate_frames, ["video_name", "frame"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Duplicate frame keys (video_name, frame): {examples}")
    video_metadata = df.groupby("video_name", sort=False).agg(
        categories=("category", "nunique"), people=("person", "nunique")
    )
    inconsistent = video_metadata[
        (video_metadata["categories"] != 1) | (video_metadata["people"] != 1)
    ]
    if not inconsistent.empty:
        raise ValueError(
            "Each video/sample must belong to exactly one category and one "
            f"person/split. Invalid examples: {inconsistent.head(10).index.tolist()}"
        )

    if allowed_person_values is not None:
        found = set(df["person"].dropna().unique())
        invalid = found - allowed_person_values
        if invalid:
            raise ValueError(
                f"Invalid values in the person/split column: {sorted(invalid)}. "
                f"Allowed values: {sorted(allowed_person_values)}"
            )
        missing_values = allowed_person_values - found
        if missing_values:
            raise ValueError(
                f"Missing required person/split values: {sorted(missing_values)}. "
                f"Found values: {sorted(found)}"
            )

    if "sign" not in df.columns:
        df["sign"] = original_categories

    return df


def prepare_landmark_dataframe(
    raw_df: pd.DataFrame,
    *,
    subset: str,
    use_imputation: bool,
    person_col: str | None = None,
    category_col: str | None = None,
    video_col: str | None = None,
    frame_col: str | None = None,
    lowercase_person: bool = False,
    allowed_person_values: set[str] | None = None,
) -> pd.DataFrame:
    """Normalize metadata, select a landmark subset, and optionally impute NaNs."""

    df = normalize_metadata(
        raw_df,
        person_col=person_col,
        category_col=category_col,
        video_col=video_col,
        frame_col=frame_col,
        lowercase_person=lowercase_person,
        allowed_person_values=allowed_person_values,
    )

    metadata_columns = [
        column for column in DEFAULT_METADATA_COLUMNS if column in df.columns
    ]
    for required in ("category", "video_name", "frame", "person"):
        if required not in metadata_columns:
            metadata_columns.append(required)

    landmark_data: dict[str, pd.Series] = {}
    missing_columns: list[str] = []
    for idx in subset_indices(subset):
        for axis in ("x", "y", "z"):
            output_column = f"{_canonical_base_name(idx)}_{axis}"
            source_column = _find_coordinate_column(
                df.columns, _expected_base_names(idx), axis
            )
            if source_column is None:
                landmark_data[output_column] = pd.Series(
                    np.nan, index=df.index, dtype="float32"
                )
                missing_columns.append(output_column)
            else:
                landmark_data[output_column] = pd.to_numeric(
                    df[source_column], errors="raise"
                )

    landmark_df = pd.DataFrame(landmark_data, index=df.index)
    result = pd.concat([df[metadata_columns], landmark_df], axis=1)
    landmark_columns = list(landmark_df.columns)

    if missing_columns:
        raise ValueError(
            f"{len(missing_columns)} required landmark coordinate columns were "
            "not found. Refusing to replace an absent schema column with zero. "
            f"Examples: {missing_columns[:10]}"
        )

    # Interpolation must follow temporal frame order within each video.
    result = result.sort_values(["video_name", "frame"]).reset_index(drop=True)
    if use_imputation:
        result = impute_by_video(result, landmark_columns)
    else:
        result[landmark_columns] = result[landmark_columns].fillna(0)

    # The Laines strategy selects 67 source landmarks and derives the 68th
    # point as the midpoint between the right and left shoulder landmarks.
    if subset == "laines":
        middle_chest = pd.DataFrame(
            {
                f"pose_middle_chest_{axis}": (
                    result[f"pose_11_{axis}"] + result[f"pose_12_{axis}"]
                )
                / 2.0
                for axis in ("x", "y", "z")
            },
            index=result.index,
        )
        result = pd.concat([result, middle_chest], axis=1)

    return result


def load_and_prepare_csv(
    data_csv: Path,
    *,
    subset: str,
    use_imputation: bool,
    person_col: str | None = None,
    category_col: str | None = None,
    video_col: str | None = None,
    frame_col: str | None = None,
    lowercase_person: bool = False,
    allowed_person_values: set[str] | None = None,
) -> pd.DataFrame:
    if not data_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {data_csv}")

    print(f"Reading source CSV: {data_csv}")
    raw_df = pd.read_csv(data_csv, na_values=["NaN", "nan", ""])
    print(f"Rows: {len(raw_df):,} | Columns: {len(raw_df.columns):,}")
    return prepare_landmark_dataframe(
        raw_df,
        subset=subset,
        use_imputation=use_imputation,
        person_col=person_col,
        category_col=category_col,
        video_col=video_col,
        frame_col=frame_col,
        lowercase_person=lowercase_person,
        allowed_person_values=allowed_person_values,
    )
