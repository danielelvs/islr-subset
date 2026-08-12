#!/usr/bin/env python3
from __future__ import annotations

"""Combine per-dataset summary CSV files into one table."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine summary CSV files from multiple datasets."
    )
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/all_datasets_summary.csv"),
    )
    args = parser.parse_args()

    files = sorted(args.reports_root.glob("*/**/*_summary.csv"))
    files += sorted(args.reports_root.glob("*/**/*_summary_outer.csv"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No summary CSV files were found under {args.reports_root}"
        )

    frames = []
    for path in files:
        dataframe = pd.read_csv(path)
        dataframe["source_file"] = str(path)
        frames.append(dataframe)

    combined = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Combined report saved to {args.output}")

    display_columns = [
        column
        for column in (
            "dataset",
            "protocol",
            "subset",
            "imputation_label",
            "f1_mean",
            "f1_std",
            "f1_count",
        )
        if column in combined.columns
    ]
    if display_columns:
        print(combined[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
