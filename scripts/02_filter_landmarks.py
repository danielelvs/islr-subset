#!/usr/bin/env python
"""
Step 2 — Filter landmarks and optionally impute missing values.

Usage:
    # All subsets for a single dataset
    python scripts/02_filter_landmarks.py -d minds

    # Specific subset, skip imputation (ablation)
    python scripts/02_filter_landmarks.py -d ufop -s 2nd --no-impute

    # All datasets, all subsets
    python scripts/02_filter_landmarks.py -d minds ufop -s all 1st 2nd laines arcanjo
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import SUBSETS, filter_and_save


def main():
    parser = argparse.ArgumentParser(description="Landmark filtering + imputation")
    parser.add_argument("-d", "--datasets",  nargs="+", default=["minds", "ufop"],
                        help="Datasets to process")
    parser.add_argument("-s", "--subsets",   nargs="+", default=list(SUBSETS.keys()),
                        help=f"Subsets: {list(SUBSETS.keys())}")
    parser.add_argument("-i", "--interim_dir",  default="data/interim",  help="Interim CSV dir")
    parser.add_argument("-o", "--processed_dir", default="data/processed", help="Output dir")
    parser.add_argument("--no-impute", action="store_true", help="Disable imputation (ablation)")
    args = parser.parse_args()

    impute = not args.no_impute

    for dataset in args.datasets:
        input_path = os.path.join(args.interim_dir, dataset, f"{dataset}_mediapipe.csv")
        if not os.path.exists(input_path):
            print(f"[SKIP] {input_path} not found.")
            continue

        for subset in args.subsets:
            suffix = "" if impute else "_no_imputation"
            out = os.path.join(args.processed_dir, dataset, f"{dataset}_{subset}{suffix}.csv")
            print(f"\n→ {dataset} / {subset} (impute={impute})")
            filter_and_save(input_path, out, subset, impute=impute, dataset_name=dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
