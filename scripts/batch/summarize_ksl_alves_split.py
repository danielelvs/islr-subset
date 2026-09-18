#!/usr/bin/env python3
"""Summarize the five KSL Alves 16/4 split folds per condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_SUBSETS = ["all", "laines", "arcanjo", "1st", "2nd"]
EXPECTED_IMPUTATIONS = ["without_imputation", "with_imputation"]
EXPECTED_FOLDS = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/ksl_alves_same_groups"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/ksl_alves_same_groups"),
    )
    args = parser.parse_args()

    result_files = sorted(args.results_dir.glob("runs/*/*/fold=*/result.json"))
    if not result_files:
        raise FileNotFoundError(
            f"No result.json files found under {args.results_dir}/runs"
        )

    rows = []
    for path in result_files:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        rows.append(
            {
                "subset": data["subset"],
                "imputation": data["imputation_label"],
                "fold": int(data["fold"]),
                "training_semantics": data.get(
                    "training_semantics", "alves_legacy_v1"
                ),
                "train_people": ",".join(data["train_people"]),
                "validation_people": ",".join(data["validation_people"]),
                "test_people": ",".join(data["test_people"]),
                "accuracy": float(data["test_accuracy"]),
                "precision": float(data["test_precision"]),
                "recall": float(data["test_recall"]),
                "f1": float(data["test_f1"]),
                "best_epoch": int(data["best_epoch"]),
                "elapsed_seconds": float(data["elapsed_seconds"]),
                "result_path": str(path),
            }
        )

    raw = pd.DataFrame(rows).sort_values(
        ["subset", "imputation", "fold"]
    ).reset_index(drop=True)

    progress_rows = []
    for subset in EXPECTED_SUBSETS:
        for imputation in EXPECTED_IMPUTATIONS:
            condition = raw[
                (raw["subset"] == subset)
                & (raw["imputation"] == imputation)
            ]
            progress_rows.append(
                {
                    "subset": subset,
                    "imputation": imputation,
                    "completed_folds": len(condition),
                    "expected_folds": EXPECTED_FOLDS,
                    "complete": len(condition) == EXPECTED_FOLDS,
                }
            )
    progress = pd.DataFrame(progress_rows)

    summary = (
        raw.groupby(["subset", "imputation"], sort=False)
        .agg(
            runs=("fold", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            elapsed_seconds_sum=("elapsed_seconds", "sum"),
        )
        .reset_index()
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.reports_dir / "raw_runs.csv"
    progress_path = args.reports_dir / "progress.csv"
    summary_path = args.reports_dir / "summary.csv"

    raw.to_csv(raw_path, index=False)
    progress.to_csv(progress_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nProgress")
    print("=" * 88)
    print(progress.to_string(index=False))

    print("\nSummary: mean ± std across the five grouped test folds")
    print("=" * 88)
    display_cols = [
        "subset",
        "imputation",
        "runs",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
    ]
    print(
        summary[display_cols].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {progress_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
