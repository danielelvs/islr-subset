#!/usr/bin/env python3
"""Summarize nested-LOPO or fixed-split batch results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    trainer = payload.get("trainer_result", {}) or {}
    metrics = payload.get("metrics", {}) or {}
    row = {
        "result_path": str(path),
        "status": payload.get("status"),
        "dataset": payload.get("dataset"),
        "protocol": payload.get("protocol"),
        "subset": payload.get("subset"),
        "subset_landmarks": payload.get("subset_landmarks"),
        "imputation": payload.get("imputation"),
        "imputation_label": payload.get("imputation_label"),
        "test_person": payload.get("test_person"),
        "val_person": payload.get("val_person"),
        "epochs": payload.get("epochs"),
        "learning_rate": payload.get("learning_rate"),
        "weight_decay": payload.get("weight_decay"),
        "batch_size": payload.get("batch_size"),
        "patience": payload.get("patience"),
        "model": payload.get("model"),
        "image_method": payload.get("image_method"),
        "seed": payload.get("seed"),
        "elapsed_seconds": as_float(payload.get("elapsed_seconds")),
        "finished_at": payload.get("finished_at"),
        "best_epoch": trainer.get("best_epoch"),
        "best_val_loss": trainer.get("best_val_loss"),
        "best_val_accuracy": trainer.get("best_val_accuracy"),
        "max_val_accuracy": trainer.get("max_val_accuracy"),
    }

    row["accuracy"] = as_float(
        metrics.get("accuracy", trainer.get("test_accuracy"))
    )
    row["precision"] = as_float(
        metrics.get("precision", trainer.get("test_precision"))
    )
    row["recall"] = as_float(metrics.get("recall", trainer.get("test_recall")))
    row["f1"] = as_float(metrics.get("f1", trainer.get("test_f1")))
    return row


def mean_std_table(dataframe: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    aggregation = {
        metric: ["mean", "std", "count"] for metric in METRIC_NAMES
    }
    aggregation["elapsed_seconds"] = ["mean", "std", "sum"]
    summary = dataframe.groupby(group_columns, dropna=False).agg(aggregation)
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
    return summary.reset_index()


def format_mean_std(mean_value: Any, std_value: Any, decimals: int = 3) -> str:
    if pd.isna(mean_value):
        return "--"
    if pd.isna(std_value):
        return f"{mean_value:.{decimals}f}"
    return f"{mean_value:.{decimals}f} ({std_value:.{decimals}f})"


def make_latex_table(summary: pd.DataFrame) -> str:
    rows = []
    table = summary.copy().sort_values(["subset", "imputation_label"])
    for _, row in table.iterrows():
        rows.append(
            {
                "Subset": row["subset"],
                "Imputation": row["imputation_label"],
                "Accuracy": format_mean_std(
                    row.get("accuracy_mean"), row.get("accuracy_std")
                ),
                "Precision": format_mean_std(
                    row.get("precision_mean"), row.get("precision_std")
                ),
                "Recall": format_mean_std(
                    row.get("recall_mean"), row.get("recall_std")
                ),
                "F1-score": format_mean_std(
                    row.get("f1_mean"), row.get("f1_std")
                ),
                "N": int(row.get("f1_count", 0))
                if not pd.isna(row.get("f1_count"))
                else 0,
            }
        )
    return pd.DataFrame(rows).to_latex(index=False, escape=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ISLR batch result.json files."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--expected-runs-per-condition", type=int, default=None)
    args = parser.parse_args()

    result_files = sorted((args.results_dir / "runs").glob("*/*/*/result.json"))
    if not result_files:
        raise FileNotFoundError(
            f"No result.json files were found under {args.results_dir / 'runs'}"
        )

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    invalid_files = []

    for path in result_files:
        try:
            rows.append(read_result(path))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            invalid_files.append(path)
            print(
                f"[WARNING] Skipping invalid result file: {path} "
                f"({type(exc).__name__}: {exc})"
            )

    if not rows:
        raise ValueError(
            "No valid result files were found. "
            "Check the experiment directory and the generated JSON files."
        )

    raw = pd.DataFrame(rows)

    if invalid_files:
        print(f"[WARNING] Invalid result files skipped: {len(invalid_files)}")


    raw = raw.sort_values(
        ["subset", "imputation_label", "test_person", "val_person"],
        na_position="last",
    ).reset_index(drop=True)

    protocols = raw["protocol"].dropna().unique().tolist()
    if len(protocols) != 1:
        raise ValueError(f"Expected one protocol, found: {protocols}")
    protocol = protocols[0]

    group_columns = [
        "dataset",
        "protocol",
        "subset",
        "subset_landmarks",
        "imputation",
        "imputation_label",
    ]
    summary_by_run = mean_std_table(raw, group_columns)

    if protocol in {"nested_lopo", "lopo_fixed_val"}:
        outer_folds = (
            raw.groupby(group_columns + ["test_person"], dropna=False)[
                METRIC_NAMES + ["elapsed_seconds"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        summary = mean_std_table(outer_folds, group_columns)
    else:
        outer_folds = raw.copy()
        summary = summary_by_run.copy()

    progress = (
        raw.groupby(["subset", "imputation_label"], dropna=False)
        .size()
        .reset_index(name="completed_runs")
    )
    if args.expected_runs_per_condition is not None:
        progress["expected_runs"] = args.expected_runs_per_condition
    else:
        progress["expected_runs"] = progress["completed_runs"].max()
    progress["progress_pct"] = (
        progress["completed_runs"] / progress["expected_runs"] * 100
    )

    pivot_f1 = summary.pivot_table(
        index="subset",
        columns="imputation_label",
        values="f1_mean",
        aggfunc="first",
    ).reset_index()

    prefix = f"{args.dataset}_{protocol}"
    paths = {
        "raw": args.reports_dir / f"{prefix}_raw_runs.csv",
        "summary_by_run": args.reports_dir / f"{prefix}_summary_by_run.csv",
        "outer_folds": args.reports_dir / f"{prefix}_outer_folds.csv",
        "summary": args.reports_dir / f"{prefix}_summary.csv",
        "progress": args.reports_dir / f"{prefix}_progress.csv",
        "pivot_f1": args.reports_dir / f"{prefix}_pivot_f1.csv",
        "latex": args.reports_dir / f"{prefix}_latex_table.tex",
    }

    raw.to_csv(paths["raw"], index=False)
    summary_by_run.to_csv(paths["summary_by_run"], index=False)
    outer_folds.to_csv(paths["outer_folds"], index=False)
    summary.to_csv(paths["summary"], index=False)
    progress.to_csv(paths["progress"], index=False)
    pivot_f1.to_csv(paths["pivot_f1"], index=False)
    paths["latex"].write_text(make_latex_table(summary), encoding="utf-8")

    print("Generated files:")
    for path in paths.values():
        print(f"- {path}")

    print("\nProgress by condition:")
    print(progress.to_string(index=False))

    print("\nMain summary:")
    display_columns = [
        "subset",
        "imputation_label",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "f1_count",
    ]
    existing = [column for column in display_columns if column in summary.columns]
    print(summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
