#!/usr/bin/env python3
"""Run one INCLUDE-50 fixed train/validation/test experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# from src.preprocessing.landmark_dataframe import effective_subset_landmark_count

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
) -> pd.DataFrame:
    if cache_enabled and cache_path.exists() and not force:
        print(f"Using preprocessed CSV cache: {cache_path}")
        return pd.read_csv(cache_path)

    processed = load_and_prepare_csv(
        data_csv,
        subset=subset,
        use_imputation=use_imputation,
        category_col="sign_id",
        video_col="sequence_id",
        frame_col="frame_id",
        person_col="split",
        lowercase_person=True,
        allowed_person_values={"train", "val", "test"},
    )
    if cache_enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(cache_path, index=False)

    videos = processed[["video_name", "person", "category"]].drop_duplicates()
    print("\nVideo distribution by split:")
    print(videos["person"].value_counts().to_string())
    print("\nNumber of classes by split:")
    print(videos.groupby("person")["category"].nunique().to_string())
    if cache_enabled:
        print(f"\nPreprocessed CSV saved to {cache_path}")
    else:
        print("\nPreprocessed data will remain in memory; no cache CSV was written.")
    print(f"Processed shape: {processed.shape}")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one INCLUDE-50 fixed split experiment."
    )
    parser.add_argument("--dataset", default="include50")
    parser.add_argument(
        "--data-csv",
        default=Path(
            "data/interim/include50_official/"
            "include50_official_mediapipe_with_split.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--results-dir",
        default=Path("experiments/include50_official_split_grid"),
        type=Path,
    )
    parser.add_argument(
        "--subset", required=True, choices=["all", "1st", "2nd", "laines", "arcanjo"]
    )
    parser.add_argument("--imputation", required=True, type=str_to_bool)
    parser.add_argument("--resume", type=str_to_bool, default=True)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--cache-preprocessed", type=str_to_bool, default=False)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--save-model", type=str_to_bool, default=False)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cuda", "cpu", "mps", "auto"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    # parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1638102311)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--image-method", default="Skeleton-DML")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        args.results_dir
        / "preprocessed"
        / f"{args.dataset}_{args.subset}_{imputation_label(args.imputation)}.csv"
    )
    trainer_tmp_dir = args.results_dir / "_trainer_tmp_outputs"

    print("\nConfiguration:")
    print(f"  dataset:       {args.dataset}")
    print(f"  data_csv:      {args.data_csv}")
    print(f"  results_dir:   {args.results_dir}")
    print(f"  subset:        {args.subset} ({effective_subset_landmark_count(args.subset)} landmarks)")
    print(f"  imputation:    {imputation_label(args.imputation)}")
    print("  protocol:      fixed train/validation/test split")
    print(f"  device:        {args.device}")
    print(f"  epochs:        {args.epochs}")
    print(f"  lr/wd/batch:   {args.lr} / {args.wd} / {args.batch_size}")
    print(f"  patience:      {args.patience}")
    print(f"  cache_csv:     {args.cache_preprocessed}")
    print(f"  save_model:    {args.save_model}")
    print(f"  seed:          {args.seed}\n")

    df = load_or_create_preprocessed(
        data_csv=args.data_csv,
        cache_path=cache_path,
        subset=args.subset,
        use_imputation=args.imputation,
        force=args.force_preprocess,
        cache_enabled=args.cache_preprocessed,
    )

    run_dir = (
        args.results_dir
        / "runs"
        / args.subset
        / imputation_label(args.imputation)
        / "fixed_split"
    )
    result_path = run_dir / "result.json"
    failed_path = run_dir / "status_failed.json"
    running_path = run_dir / "status_running.json"
    checkpoint_path = run_dir / "best_model.pth"

    if args.resume and result_path.exists() and not args.force_run:
        print(f"Result already exists; skipping run: {result_path}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    failed_path.unlink(missing_ok=True)
    save_json(
        running_path,
        {
            "status": "running",
            "dataset": args.dataset,
            "protocol": "fixed_split",
            "subset": args.subset,
            "imputation": args.imputation,
            "started_at": datetime.now().isoformat(),
        },
    )

    reference = (
        f"dataset={args.dataset}__protocol=fixed_split__subset={args.subset}"
        f"__imputation={imputation_label(args.imputation)}__model={args.model}"
        f"__repr={args.image_method.replace('/', '-')}__epochs={args.epochs}"
        f"__lr={args.lr:g}__wd={args.wd:g}__batch={args.batch_size}"
    )

    trainer_config = {
        "dataset_name": args.dataset,
        "ref": reference,
        "seed": args.seed,
        "validate_people": ["val"],
        "test_people": ["test"],
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
            # mirror_chance=0.3
        },
        "output_dir": str(trainer_tmp_dir),
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
            "protocol": "fixed_split",
            "subset": args.subset,
            "subset_landmarks": effective_subset_landmark_count(args.subset),
            "imputation": args.imputation,
            "imputation_label": imputation_label(args.imputation),
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
        print(f"Result saved to {result_path}")
        if args.save_model:
            print(f"Checkpoint saved to {checkpoint_path}")
        print(json.dumps(payload["metrics"], indent=2))
    except Exception as error:
        elapsed_seconds = time.perf_counter() - start_time
        save_json(
            failed_path,
            {
                "status": "failed",
                "dataset": args.dataset,
                "protocol": "fixed_split",
                "subset": args.subset,
                "imputation": args.imputation,
                "elapsed_seconds": elapsed_seconds,
                "failed_at": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        running_path.unlink(missing_ok=True)
        print("[ERROR] Run failed")
        print(f"{type(error).__name__}: {error}")
        raise
    finally:
        temporary_run_dir = trainer_tmp_dir / reference
        if temporary_run_dir.exists():
            shutil.rmtree(temporary_run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
