#!/usr/bin/env python3
"""
Summarize KSL nested LOPO batch results.

Expected input:
  experiments/ksl_nested_lopo_resume_grid/runs/<subset>/<imputation>/test=...__val=.../result.json

Outputs:
  reports/ksl/ksl_nested_lopo_raw_runs.csv
  reports/ksl/ksl_nested_lopo_summary_by_run.csv
  reports/ksl/ksl_nested_lopo_outer_folds.csv
  reports/ksl/ksl_nested_lopo_summary_outer.csv
  reports/ksl/ksl_nested_lopo_pivot_f1.csv
  reports/ksl/ksl_nested_lopo_latex_table.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_result(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)

    trainer = payload.get("trainer_result", {}) or {}
    metrics = payload.get("metrics", {}) or {}

    row: dict[str, Any] = {
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
        "flip": payload.get("flip"),
        "rotation_sigma": payload.get("rotation_sigma"),
        "zoom_sigma": payload.get("zoom_sigma"),
        "translate_x_sigma": payload.get("translate_x_sigma"),
        "model": payload.get("model"),
        "image_method": payload.get("image_method"),
        "seed": payload.get("seed"),
        "elapsed_seconds": as_float(payload.get("elapsed_seconds")),
        "finished_at": payload.get("finished_at"),
    }

    # Prefer explicit metrics block, fall back to trainer_result keys.
    row["accuracy"] = as_float(metrics.get("accuracy", trainer.get("test_accuracy")))
    row["precision"] = as_float(metrics.get("precision", trainer.get("test_precision")))
    row["recall"] = as_float(metrics.get("recall", trainer.get("test_recall")))
    row["f1"] = as_float(metrics.get("f1", trainer.get("test_f1")))

    # Optional training/validation metrics if Trainer produced them.
    optional_keys = [
        "best_epoch",
        "best_val_loss",
        "best_val_accuracy",
        "best_val_precision",
        "best_val_recall",
        "best_val_f1",
        "train_loss",
        "val_loss",
    ]
    for key in optional_keys:
        row[key] = trainer.get(key)

    return row


def mean_std_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg_spec: dict[str, list[str]] = {metric: ["mean", "std", "count"] for metric in METRIC_NAMES}
    agg_spec["elapsed_seconds"] = ["mean", "std", "sum"]
    summary = df.groupby(group_cols, dropna=False).agg(agg_spec)
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary = summary.reset_index()
    return summary


def format_mean_std(mean_value: Any, std_value: Any, decimals: int = 3) -> str:
    if pd.isna(mean_value):
        return "--"
    if pd.isna(std_value):
        return f"{mean_value:.{decimals}f}"
    return f"{mean_value:.{decimals}f} ({std_value:.{decimals}f})"


def make_latex_table(summary_outer: pd.DataFrame) -> str:
    rows = []
    table_df = summary_outer.copy()
    table_df["imputation_label"] = table_df["imputation_label"].fillna(table_df["imputation"].astype(str))
    table_df = table_df.sort_values(["subset", "imputation_label"])

    for _, row in table_df.iterrows():
        rows.append(
            {
                "Subset": row["subset"],
                "Imputation": row["imputation_label"],
                "Accuracy": format_mean_std(row.get("accuracy_mean"), row.get("accuracy_std")),
                "Precision": format_mean_std(row.get("precision_mean"), row.get("precision_std")),
                "Recall": format_mean_std(row.get("recall_mean"), row.get("recall_std")),
                "F1-score": format_mean_std(row.get("f1_mean"), row.get("f1_std")),
                "N": int(row.get("f1_count", 0)) if not pd.isna(row.get("f1_count")) else 0,
            }
        )

    out = pd.DataFrame(rows)
    return out.to_latex(index=False, escape=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KSL nested LOPO results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/ksl_nested_lopo_resume_grid"),
        help="Directory containing runs/ with result.json files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/ksl"),
        help="Directory where CSV/TEX summaries will be written.",
    )
    parser.add_argument("--expected-runs-per-condition", type=int, default=380)
    args = parser.parse_args()

    result_files = sorted((args.results_dir / "runs").glob("*/*/*/result.json"))
    if not result_files:
        raise FileNotFoundError(f"Não encontrei result.json em: {args.results_dir / 'runs'}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.DataFrame([read_result(path) for path in result_files])
    raw = raw.sort_values(["subset", "imputation_label", "test_person", "val_person"]).reset_index(drop=True)

    # Per-run summary: every test/validation pair counts as one run.
    summary_by_run = mean_std_table(raw, ["dataset", "protocol", "subset", "subset_landmarks", "imputation", "imputation_label"])

    # Outer-fold summary: first average over validation people for each test person,
    # then compute mean/std across test people. This avoids treating the same test person
    # as 19 fully independent samples.
    outer_folds = (
        raw.groupby([
            "dataset",
            "protocol",
            "subset",
            "subset_landmarks",
            "imputation",
            "imputation_label",
            "test_person",
        ], dropna=False)[METRIC_NAMES + ["elapsed_seconds"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary_outer = mean_std_table(
        outer_folds,
        ["dataset", "protocol", "subset", "subset_landmarks", "imputation", "imputation_label"],
    )

    # Completion status by condition.
    completed = (
        raw.groupby(["subset", "imputation_label"], dropna=False)
        .size()
        .reset_index(name="completed_runs")
    )
    completed["expected_runs"] = args.expected_runs_per_condition
    completed["progress_pct"] = completed["completed_runs"] / completed["expected_runs"] * 100

    # Handy pivot focused on F1.
    pivot_f1 = summary_outer.pivot_table(
        index="subset",
        columns="imputation_label",
        values="f1_mean",
        aggfunc="first",
    ).reset_index()

    raw_path = args.reports_dir / "ksl_nested_lopo_raw_runs.csv"
    summary_run_path = args.reports_dir / "ksl_nested_lopo_summary_by_run.csv"
    outer_path = args.reports_dir / "ksl_nested_lopo_outer_folds.csv"
    summary_outer_path = args.reports_dir / "ksl_nested_lopo_summary_outer.csv"
    progress_path = args.reports_dir / "ksl_nested_lopo_progress.csv"
    pivot_path = args.reports_dir / "ksl_nested_lopo_pivot_f1.csv"
    latex_path = args.reports_dir / "ksl_nested_lopo_latex_table.tex"

    raw.to_csv(raw_path, index=False)
    summary_by_run.to_csv(summary_run_path, index=False)
    outer_folds.to_csv(outer_path, index=False)
    summary_outer.to_csv(summary_outer_path, index=False)
    completed.to_csv(progress_path, index=False)
    pivot_f1.to_csv(pivot_path, index=False)
    latex_path.write_text(make_latex_table(summary_outer))

    print("Arquivos gerados:")
    for path in [raw_path, summary_run_path, outer_path, summary_outer_path, progress_path, pivot_path, latex_path]:
        print(f"- {path}")

    print("\nProgresso por condição:")
    print(completed.to_string(index=False))

    print("\nResumo principal: média ± desvio por outer fold/test_person")
    display_cols = [
        "subset",
        "imputation_label",
        "accuracy_mean", "accuracy_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std",
        "f1_count",
    ]
    existing = [c for c in display_cols if c in summary_outer.columns]
    print(summary_outer[existing].to_string(index=False))


if __name__ == "__main__":
    main()
