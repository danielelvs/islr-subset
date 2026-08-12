#!/usr/bin/env python3
"""Run one nested-LOPO subset/imputation condition with fold-level resume."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.landmark_dataframe import (
    effective_subset_landmark_count,  # noqa: E402
    imputation_label,
    load_and_prepare_csv,
    subset_indices,
)
from training.trainer import Trainer  # noqa: E402


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "with"}:
        return True
    if normalized in {"false", "0", "no", "n", "without"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid Boolean value: {value}")


def parse_max_runs(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", "all", ""}:
        return None
    parsed = int(normalized)
    if parsed < 1:
        raise argparse.ArgumentTypeError("max-runs must be at least 1")
    return parsed


def sanitize_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value))
    return text.strip("-") or "unknown"


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)
    os.replace(temporary_path, path)


def load_or_create_preprocessed(
    *,
    data_csv: Path,
    cache_path: Path,
    subset: str,
    use_imputation: bool,
    force: bool,
    cache_enabled: bool,
    person_col: str | None,
    category_col: str | None,
    video_col: str | None,
    frame_col: str | None,
) -> pd.DataFrame:
    if cache_enabled and cache_path.exists() and not force:
        print(f"Using preprocessed CSV cache: {cache_path}")
        return pd.read_csv(cache_path)

    processed = load_and_prepare_csv(
        data_csv,
        subset=subset,
        use_imputation=use_imputation,
        person_col=person_col,
        category_col=category_col,
        video_col=video_col,
        frame_col=frame_col,
    )
    if cache_enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(cache_path, index=False)
        print(f"Preprocessed CSV saved to {cache_path}")
    else:
        print("Preprocessed data will remain in memory; no cache CSV was written.")
    print(f"Processed shape: {processed.shape}")
    return processed


def build_run_plan(
    df: pd.DataFrame,
    *,
    protocol: str,
    fixed_validation_person: str | None,
) -> list[tuple[Any, Any]]:
    people = sorted(df["person"].dropna().unique(), key=lambda value: str(value))
    if len(people) < 3:
        raise ValueError(
            "LOPO training requires at least three people/groups so train, "
            "validation, and test partitions are all non-empty. "
            f"Found: {people}"
        )

    if protocol == "nested_lopo":
        return [
            (test_person, validation_person)
            for test_person in people
            for validation_person in people
            if validation_person != test_person
        ]

    if protocol == "lopo_fixed_val":
        if fixed_validation_person is None:
            raise ValueError(
                "--fixed-val-person is required with protocol lopo_fixed_val"
            )
        matching = [
            person
            for person in people
            if str(person) == str(fixed_validation_person)
        ]
        if not matching:
            raise ValueError(
                f"Fixed validation person '{fixed_validation_person}' was not found. "
                f"Available values: {people}"
            )
        validation_person = matching[0]
        return [
            (test_person, validation_person)
            for test_person in people
            if test_person != validation_person
        ]

    raise ValueError(f"Unsupported protocol: {protocol}")


def run_directory(
    results_dir: Path,
    subset: str,
    use_imputation: bool,
    test_person: Any,
    validation_person: Any,
) -> Path:
    return (
        results_dir
        / "runs"
        / subset
        / imputation_label(use_imputation)
        / f"test={sanitize_id(test_person)}__val={sanitize_id(validation_person)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nested-LOPO ISLR experiments with fold-level resume."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-csv", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument(
        "--subset", required=True, choices=["all", "1st", "2nd", "laines", "arcanjo"]
    )
    parser.add_argument("--imputation", required=True, type=str_to_bool)
    parser.add_argument("--max-runs", default="none")
    parser.add_argument("--resume", type=str_to_bool, default=True)
    parser.add_argument("--skip-failed", type=str_to_bool, default=False)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--cache-preprocessed", type=str_to_bool, default=False)
    parser.add_argument("--save-model", type=str_to_bool, default=False)
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu", "mps", "auto"]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--image-method", default="Skeleton-DML")
    parser.add_argument(
        "--protocol",
        default="nested_lopo",
        choices=["nested_lopo", "lopo_fixed_val"],
    )
    parser.add_argument("--fixed-val-person", default=None)
    parser.add_argument("--person-col", default=None)
    parser.add_argument("--category-col", default=None)
    parser.add_argument("--video-col", default=None)
    parser.add_argument("--frame-col", default=None)
    args = parser.parse_args()

    max_runs = parse_max_runs(args.max_runs)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        args.results_dir
        / "preprocessed"
        / f"{args.dataset}_{args.subset}_{imputation_label(args.imputation)}.csv"
    )
    temporary_output_dir = args.results_dir / "_trainer_tmp_outputs"

    print("\nConfiguration:")
    print(f"  dataset:       {args.dataset}")
    print(f"  data_csv:      {args.data_csv}")
    print(f"  results_dir:   {args.results_dir}")
    print(f"  subset:        {args.subset} ({effective_subset_landmark_count(args.subset)} landmarks)")
    print(f"  imputation:    {imputation_label(args.imputation)}")
    print(f"  protocol:      {args.protocol}")
    print(f"  device:        {args.device}")
    print(f"  epochs:        {args.epochs}")
    print(f"  lr/wd/batch:   {args.lr} / {args.wd} / {args.batch_size}")
    print(f"  patience:      {args.patience}")
    print(f"  max_runs:      {max_runs}")
    print(f"  cache_csv:     {args.cache_preprocessed}")
    print(f"  save_model:    {args.save_model}")
    print(f"  resume:        {args.resume}\n")

    df = load_or_create_preprocessed(
        data_csv=args.data_csv,
        cache_path=cache_path,
        subset=args.subset,
        use_imputation=args.imputation,
        force=args.force_preprocess,
        cache_enabled=args.cache_preprocessed,
        person_col=args.person_col,
        category_col=args.category_col,
        video_col=args.video_col,
        frame_col=args.frame_col,
    )
    run_plan = build_run_plan(
        df,
        protocol=args.protocol,
        fixed_validation_person=args.fixed_val_person,
    )

    def result_exists(test_person: Any, validation_person: Any) -> bool:
        return (
            run_directory(
                args.results_dir,
                args.subset,
                args.imputation,
                test_person,
                validation_person,
            )
            / "result.json"
        ).exists()

    completed_before = sum(
        result_exists(test_person, validation_person)
        for test_person, validation_person in run_plan
    )
    total = len(run_plan)
    print(
        f"Plan: {total} runs | completed={completed_before} | "
        f"pending={total - completed_before}\n"
    )

    executed_now = 0
    skipped_completed = 0
    skipped_failed = 0
    failed_now = 0

    progress = tqdm(run_plan, total=total, desc=args.protocol)
    for test_person, validation_person in progress:
        description = (
            f"{args.subset} | {imputation_label(args.imputation)} | "
            f"test={test_person} | val={validation_person}"
        )
        progress.set_description(description)

        current_run_dir = run_directory(
            args.results_dir,
            args.subset,
            args.imputation,
            test_person,
            validation_person,
        )
        result_path = current_run_dir / "result.json"
        failed_path = current_run_dir / "status_failed.json"
        running_path = current_run_dir / "status_running.json"
        checkpoint_path = current_run_dir / "best_model.pth"

        if args.resume and result_path.exists():
            skipped_completed += 1
            continue
        if args.skip_failed and failed_path.exists():
            skipped_failed += 1
            continue
        if max_runs is not None and executed_now >= max_runs:
            break

        current_run_dir.mkdir(parents=True, exist_ok=True)
        failed_path.unlink(missing_ok=True)
        save_json(
            running_path,
            {
                "status": "running",
                "dataset": args.dataset,
                "protocol": args.protocol,
                "subset": args.subset,
                "imputation": args.imputation,
                "test_person": test_person,
                "val_person": validation_person,
                "started_at": datetime.now().isoformat(),
            },
        )

        reference = (
            f"dataset={args.dataset}__protocol={args.protocol}__subset={args.subset}"
            f"__imputation={imputation_label(args.imputation)}__model={args.model}"
            f"__repr={args.image_method.replace('/', '-')}__epochs={args.epochs}"
            f"__lr={args.lr:g}__wd={args.wd:g}__batch={args.batch_size}"
            f"__test={sanitize_id(test_person)}__val={sanitize_id(validation_person)}"
        )

        trainer_config = {
            "dataset_name": args.dataset,
            "ref": reference,
            "seed": args.seed,
            "validate_people": [validation_person],
            "test_people": [test_person],
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "image_method": args.image_method,
            "model": args.model,
            "device": args.device,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "augment_cfg": {
                "rotation_sigma": 12,
                "zoom_sigma": 0.1,
                "translate_x_sigma": 0.06,
                "translate_y_sigma": 0.0,
                "translate_z_sigma": 0.0,
                "horizontal_flip_prob": 0.5,
            },
            "output_dir": str(temporary_output_dir),
            "save_model": args.save_model,
            "model_output_path": str(checkpoint_path),
        }

        start_time = time.perf_counter()
        try:
            trainer_result = Trainer(trainer_config).run(df)
            elapsed_seconds = time.perf_counter() - start_time
            payload = {
                "status": "completed",
                "dataset": args.dataset,
                "protocol": args.protocol,
                "subset": args.subset,
                "subset_landmarks": effective_subset_landmark_count(args.subset),
                "imputation": args.imputation,
                "imputation_label": imputation_label(args.imputation),
                "test_person": test_person,
                "val_person": validation_person,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": args.wd,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "model": args.model,
                "image_method": args.image_method,
                "device": args.device,
                "seed": args.seed,
                "elapsed_seconds": elapsed_seconds,
                "checkpoint_path": str(checkpoint_path) if args.save_model else None,
                "finished_at": datetime.now().isoformat(),
                "metrics": {
                    "accuracy": trainer_result.get("test_accuracy"),
                    "precision": trainer_result.get("test_precision"),
                    "recall": trainer_result.get("test_recall"),
                    "f1": trainer_result.get("test_f1"),
                },
                "trainer_result": trainer_result,
            }
            save_json(result_path, payload)
            running_path.unlink(missing_ok=True)
            executed_now += 1
            print(f"Result saved to {result_path}")
        except Exception as error:
            elapsed_seconds = time.perf_counter() - start_time
            failed_now += 1
            save_json(
                failed_path,
                {
                    "status": "failed",
                    "dataset": args.dataset,
                    "protocol": args.protocol,
                    "subset": args.subset,
                    "imputation": args.imputation,
                    "test_person": test_person,
                    "val_person": validation_person,
                    "elapsed_seconds": elapsed_seconds,
                    "failed_at": datetime.now().isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            running_path.unlink(missing_ok=True)
            print(f"[ERROR] Run failed: {description}")
            print(f"        {type(error).__name__}: {error}")
        finally:
            temporary_run_dir = temporary_output_dir / reference
            if temporary_run_dir.exists():
                shutil.rmtree(temporary_run_dir, ignore_errors=True)

    completed_after = sum(
        result_exists(test_person, validation_person)
        for test_person, validation_person in run_plan
    )
    summary = {
        "executed_now": executed_now,
        "skipped_completed": skipped_completed,
        "skipped_failed": skipped_failed,
        "failed_now": failed_now,
        "max_runs": max_runs,
        "total": total,
        "completed": completed_after,
        "pending": total - completed_after,
    }
    print(json.dumps(summary, indent=2))
    print(
        f"Condition progress: {completed_after}/{total} "
        f"({completed_after / total * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
