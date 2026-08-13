#!/usr/bin/env python3
"""Run KSL with five disjoint 12/4/4 signer folds."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Project root not found. Run this script from the repository root or "
        "ensure that a src/ directory exists."
    )


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.landmark_dataframe import (  # noqa: E402
    effective_subset_landmark_count,
    imputation_label,
    load_and_prepare_csv,
)
from training.trainer import Trainer  # noqa: E402


GROUPS: list[list[str]] = [
    ["0", "1", "2", "3"],
    ["4", "5", "6", "7"],
    ["8", "9", "10", "11"],
    ["12", "13", "14", "15"],
    ["16", "17", "18", "19"],
]

FOLDS: list[dict[str, Any]] = [
    {"fold": 1, "test": GROUPS[0], "val": GROUPS[1]},
    {"fold": 2, "test": GROUPS[1], "val": GROUPS[2]},
    {"fold": 3, "test": GROUPS[2], "val": GROUPS[3]},
    {"fold": 4, "test": GROUPS[3], "val": GROUPS[4]},
    {"fold": 5, "test": GROUPS[4], "val": GROUPS[0]},
]

EXPECTED_SIGNERS = {str(i) for i in range(20)}


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "with"}:
        return True
    if normalized in {"false", "0", "no", "n", "without"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid Boolean value: {value}")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)
    os.replace(temporary_path, path)


def sorted_signers(values: set[str] | list[str]) -> list[str]:
    return sorted((str(v) for v in values), key=lambda value: int(value))


def load_or_create_preprocessed(
    *,
    data_csv: Path,
    cache_path: Path,
    subset: str,
    use_imputation: bool,
    force: bool,
    cache_enabled: bool,
) -> pd.DataFrame:
    if cache_enabled and cache_path.exists() and not force:
        print(f"Using preprocessed CSV cache: {cache_path}")
        return pd.read_csv(cache_path)

    processed = load_and_prepare_csv(
        data_csv,
        subset=subset,
        use_imputation=use_imputation,
    )

    processed["person"] = processed["person"].astype(str).str.strip()

    if cache_enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(cache_path, index=False)
        print(f"Preprocessed CSV saved to: {cache_path}")
    else:
        print("Preprocessed data will remain in memory; no cache CSV was written.")

    return processed


def validate_dataset(df: pd.DataFrame) -> None:
    videos = df[["video_name", "person", "category"]].drop_duplicates()
    found_signers = set(videos["person"].astype(str))

    print("\nKSL dataset diagnostic")
    print("-" * 72)
    print(f"Videos:          {videos['video_name'].nunique()}")
    print(f"Classes:         {videos['category'].nunique()}")
    print(f"Signers:         {len(found_signers)}")
    print(f"Signer IDs:      {sorted_signers(found_signers)}")

    if found_signers != EXPECTED_SIGNERS:
        missing = sorted_signers(EXPECTED_SIGNERS - found_signers)
        extra = sorted_signers(found_signers - EXPECTED_SIGNERS)
        raise ValueError(
            "The grouped 12/4/4 protocol expects signer IDs 0..19. "
            f"Missing={missing}; extra={extra}"
        )

    if videos["category"].nunique() != 67:
        print(
            f"[WARNING] Expected 67 KSL classes for the current dataset version, "
            f"but found {videos['category'].nunique()}."
        )

    if videos["video_name"].nunique() != 1229:
        print(
            f"[WARNING] Expected 1,229 KSL videos for the current dataset version, "
            f"but found {videos['video_name'].nunique()}."
        )


def fold_partition(fold_spec: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    test_people = [str(v) for v in fold_spec["test"]]
    val_people = [str(v) for v in fold_spec["val"]]
    train_people = sorted_signers(
        EXPECTED_SIGNERS - set(test_people) - set(val_people)
    )

    train_set = set(train_people)
    val_set = set(val_people)
    test_set = set(test_people)

    if len(train_people) != 12 or len(val_people) != 4 or len(test_people) != 4:
        raise RuntimeError(
            f"Invalid split sizes in fold {fold_spec['fold']}: "
            f"train={len(train_people)}, val={len(val_people)}, test={len(test_people)}"
        )
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError(f"Signer leakage detected in fold {fold_spec['fold']}.")
    if train_set | val_set | test_set != EXPECTED_SIGNERS:
        raise RuntimeError(f"Fold {fold_spec['fold']} does not cover all 20 signers.")

    return train_people, val_people, test_people


def print_fold_distribution(
    df: pd.DataFrame,
    *,
    fold: int,
    train_people: list[str],
    val_people: list[str],
    test_people: list[str],
) -> None:
    videos = df[["video_name", "person", "category"]].drop_duplicates()

    def describe(name: str, people: list[str]) -> None:
        part = videos[videos["person"].isin(people)]
        print(
            f"  {name:<5} signers={people} | "
            f"videos={part['video_name'].nunique():4d} | "
            f"classes={part['category'].nunique():2d}"
        )

    print(f"\nFold {fold}")
    describe("train", train_people)
    describe("val", val_people)
    describe("test", test_people)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run KSL using five disjoint 12-train/4-val/4-test signer folds."
    )
    parser.add_argument("--dataset", default="ksl")
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=Path("data/interim/ksl/ksl_mediapipe.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/ksl_grouped_12_4_4"),
    )
    parser.add_argument(
        "--subset",
        required=True,
        choices=["all", "1st", "2nd", "laines", "arcanjo"],
    )
    parser.add_argument("--imputation", required=True, type=str_to_bool)

    parser.add_argument("--resume", type=str_to_bool, default=True)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--cache-preprocessed", type=str_to_bool, default=False)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--save-model", type=str_to_bool, default=False)

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps", "auto"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1638102311)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--image-method", default="Skeleton-DML")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="0 runs all five folds; use 1 for a smoke test.",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    cache_path = (
        args.results_dir
        / "preprocessed"
        / f"ksl_{args.subset}_{imputation_label(args.imputation)}.csv"
    )
    trainer_tmp_dir = args.results_dir / "_trainer_tmp_outputs"

    print("\nConfiguration")
    print("=" * 72)
    print(f"dataset:          {args.dataset}")
    print(f"data_csv:         {args.data_csv}")
    print(f"results_dir:      {args.results_dir}")
    print(
        f"subset:           {args.subset} "
        f"({effective_subset_landmark_count(args.subset)} effective landmarks)"
    )
    print(f"imputation:       {imputation_label(args.imputation)}")
    print("protocol:         grouped signer-independent 12/4/4")
    print("folds:            5")
    print(f"device:           {args.device}")
    print(f"epochs:           {args.epochs}")
    print(f"lr:               {args.lr}")
    print(f"weight_decay:     {args.wd}")
    print(f"batch_size:       {args.batch_size}")
    print(f"patience:         {args.patience}")
    print(f"seed:             {args.seed}")
    print(f"resume:           {args.resume}")
    print(f"save_model:       {args.save_model}")
    print(f"cache_csv:        {args.cache_preprocessed}")
    print("=" * 72)

    df = load_or_create_preprocessed(
        data_csv=args.data_csv,
        cache_path=cache_path,
        subset=args.subset,
        use_imputation=args.imputation,
        force=args.force_preprocess,
        cache_enabled=args.cache_preprocessed,
    )
    validate_dataset(df)

    completed_this_invocation = 0

    for fold_spec in FOLDS:
        fold = int(fold_spec["fold"])
        train_people, val_people, test_people = fold_partition(fold_spec)

        print_fold_distribution(
            df,
            fold=fold,
            train_people=train_people,
            val_people=val_people,
            test_people=test_people,
        )

        fold_name = (
            f"fold={fold:02d}"
            f"__test={'-'.join(test_people)}"
            f"__val={'-'.join(val_people)}"
        )
        run_dir = (
            args.results_dir
            / "runs"
            / args.subset
            / imputation_label(args.imputation)
            / fold_name
        )
        result_path = run_dir / "result.json"
        running_path = run_dir / "status_running.json"
        failed_path = run_dir / "status_failed.json"
        checkpoint_path = run_dir / "best_model.pth"

        if args.resume and result_path.exists() and not args.force_run:
            print(f"Result already exists; skipping: {result_path}")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        failed_path.unlink(missing_ok=True)

        save_json(
            running_path,
            {
                "status": "running",
                "dataset": args.dataset,
                "protocol": "grouped_12_4_4",
                "fold": fold,
                "subset": args.subset,
                "imputation": args.imputation,
                "train_people": train_people,
                "validation_people": val_people,
                "test_people": test_people,
                "started_at": datetime.now().isoformat(),
            },
        )

        reference = (
            f"dataset={args.dataset}"
            f"__protocol=grouped_12_4_4"
            f"__subset={args.subset}"
            f"__imputation={imputation_label(args.imputation)}"
            f"__fold={fold:02d}"
            f"__model={args.model}"
            f"__repr={args.image_method.replace('/', '-')}"
            f"__epochs={args.epochs}"
            f"__lr={args.lr:g}"
            f"__wd={args.wd:g}"
            f"__batch={args.batch_size}"
        )

        trainer_config = {
            "dataset_name": args.dataset,
            "ref": reference,
            "seed": args.seed,
            "validate_people": val_people,
            "test_people": test_people,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "image_method": args.image_method,
            "model": args.model,
            "device": args.device,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "output_dir": trainer_tmp_dir,
            "save_model": args.save_model,
            "model_output_path": checkpoint_path if args.save_model else None,
        }

        started = time.perf_counter()

        try:
            trainer_result = Trainer(trainer_config).run(df)
            elapsed_seconds = time.perf_counter() - started

            result = {
                "status": "completed",
                "dataset": args.dataset,
                "protocol": "grouped_12_4_4",
                "fold": fold,
                "subset": args.subset,
                "subset_landmarks": effective_subset_landmark_count(args.subset),
                "imputation": args.imputation,
                "imputation_label": imputation_label(args.imputation),
                "train_people": train_people,
                "validation_people": val_people,
                "test_people": test_people,
                "n_train_signers": len(train_people),
                "n_validation_signers": len(val_people),
                "n_test_signers": len(test_people),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.wd,
                "patience": args.patience,
                "seed": args.seed,
                "model": args.model,
                "image_method": args.image_method,
                "elapsed_seconds": elapsed_seconds,
                "test_accuracy": trainer_result["test_accuracy"],
                "test_precision": trainer_result["test_precision"],
                "test_recall": trainer_result["test_recall"],
                "test_f1": trainer_result["test_f1"],
                "best_epoch": trainer_result["best_epoch"],
                "best_val_accuracy": trainer_result["best_val_accuracy"],
                "best_val_loss": trainer_result["best_val_loss"],
                "trainer_result": trainer_result,
                "completed_at": datetime.now().isoformat(),
            }
            save_json(result_path, result)
            running_path.unlink(missing_ok=True)

            print(
                f"\nCompleted fold {fold}: "
                f"acc={result['test_accuracy']:.4f}, "
                f"f1={result['test_f1']:.4f}"
            )
            print(f"Saved: {result_path}")

        except Exception as exc:
            elapsed_seconds = time.perf_counter() - started
            save_json(
                failed_path,
                {
                    "status": "failed",
                    "dataset": args.dataset,
                    "protocol": "grouped_12_4_4",
                    "fold": fold,
                    "subset": args.subset,
                    "imputation": args.imputation,
                    "train_people": train_people,
                    "validation_people": val_people,
                    "test_people": test_people,
                    "elapsed_seconds": elapsed_seconds,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at": datetime.now().isoformat(),
                },
            )
            running_path.unlink(missing_ok=True)
            raise

        completed_this_invocation += 1
        if args.max_runs > 0 and completed_this_invocation >= args.max_runs:
            print(f"\nReached --max-runs={args.max_runs}; stopping.")
            break


if __name__ == "__main__":
    main()
