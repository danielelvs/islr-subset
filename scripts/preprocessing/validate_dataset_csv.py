#!/usr/bin/env python3
from __future__ import annotations

"""Validate a landmark CSV before extraction-independent batch training."""

import argparse
from pathlib import Path

import pandas as pd

META_ALIASES = {
    "person": ["person", "signer", "interpreter", "participant_id", "split", "signer_id"],
    "category": ["category", "sign_id", "label", "class", "class_id"],
    "video_name": ["video_name", "sequence_id", "path", "file", "filename", "sample_id"],
    "frame": ["frame", "frame_id"],
}


def find_any(columns, names):
    return next((name for name in names if name in columns), None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an ISLR landmark CSV before batch training."
    )
    parser.add_argument("--data-csv", required=True, type=Path)
    parser.add_argument("--nrows", type=int, default=5000)
    parser.add_argument("--person-col", default=None)
    parser.add_argument("--category-col", default=None)
    parser.add_argument("--video-col", default=None)
    parser.add_argument("--frame-col", default=None)
    parser.add_argument("--expected-coordinate-columns", type=int, default=1629)
    args = parser.parse_args()

    if not args.data_csv.exists():
        raise FileNotFoundError(args.data_csv)

    dataframe = pd.read_csv(args.data_csv, nrows=args.nrows)
    print(f"File: {args.data_csv}")
    print(f"Sample: {len(dataframe):,} rows | {len(dataframe.columns):,} columns")

    missing_metadata: list[str] = []
    configured = {
        "person": args.person_col,
        "category": args.category_col,
        "video_name": args.video_col,
        "frame": args.frame_col,
    }
    for canonical_name, aliases in META_ALIASES.items():
        explicit = configured[canonical_name]
        found = explicit if explicit in dataframe.columns else None
        if explicit is None:
            found = find_any(dataframe.columns, aliases)
        print(f"{canonical_name:12s}: {'OK -> ' + found if found else 'NOT FOUND'}")
        if found is None:
            missing_metadata.append(canonical_name)

    landmark_columns = [
        column
        for column in dataframe.columns
        if column.endswith(("_x", "_y", "_z", ".x", ".y", ".z"))
    ]
    print(f"landmarks   : {len(landmark_columns)} coordinate columns detected")
    if landmark_columns:
        print(f"Examples: {landmark_columns[:12]}")

    coordinate_count_ok = len(landmark_columns) == args.expected_coordinate_columns
    if not coordinate_count_ok:
        print(
            f"Expected {args.expected_coordinate_columns} coordinate columns, "
            f"found {len(landmark_columns)}."
        )

    if missing_metadata or not coordinate_count_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
