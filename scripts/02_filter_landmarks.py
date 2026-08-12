#!/usr/bin/env python3
"""Select landmark subsets and optionally impute missing coordinates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import SUBSETS, filter_and_save


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter MediaPipe landmark CSV files by subset."
    )
    parser.add_argument(
        "-d", "--datasets", nargs="+", default=["minds", "ufop"]
    )
    parser.add_argument(
        "-s", "--subsets", nargs="+", default=list(SUBSETS)
    )
    parser.add_argument("-i", "--interim-dir", default="data/interim")
    parser.add_argument("-o", "--processed-dir", default="data/processed")
    parser.add_argument("--no-imputation", action="store_true")
    args = parser.parse_args()

    use_imputation = not args.no_imputation
    for dataset in args.datasets:
        input_path = (
            Path(args.interim_dir) / dataset / f"{dataset}_mediapipe.csv"
        )
        if not input_path.exists():
            print(f"[SKIP] Input CSV not found: {input_path}")
            continue

        for subset in args.subsets:
            suffix = "" if use_imputation else "_no_imputation"
            output_path = (
                Path(args.processed_dir)
                / dataset
                / f"{dataset}_{subset}{suffix}.csv"
            )
            print(
                f"\nProcessing dataset={dataset}, subset={subset}, "
                f"imputation={use_imputation}"
            )
            filter_and_save(
                str(input_path),
                str(output_path),
                subset,
                impute=use_imputation,
                dataset_name=dataset,
            )

    print("\nFiltering completed.")


if __name__ == "__main__":
    main()
