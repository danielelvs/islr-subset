#!/usr/bin/env python3
from __future__ import annotations

"""Validate a landmark CSV before extraction-independent batch training."""

import argparse
from pathlib import Path

import pandas as pd

META_ALIASES = {
    "person": ["person", "signer", "interpreter", "participant_id", "split"],
    "category": ["category", "sign_id", "label", "class"],
    "video_name": ["video_name", "sequence_id", "path", "file", "filename"],
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
    args = parser.parse_args()

    if not args.data_csv.exists():
        raise FileNotFoundError(args.data_csv)

    dataframe = pd.read_csv(args.data_csv, nrows=args.nrows)
    print(f"File: {args.data_csv}")
    print(f"Sample: {len(dataframe):,} rows | {len(dataframe.columns):,} columns")

    missing_metadata: list[str] = []
    for canonical_name, aliases in META_ALIASES.items():
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

    if missing_metadata or not landmark_columns:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
