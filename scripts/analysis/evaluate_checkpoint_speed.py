#!/usr/bin/env python3
"""Evaluate checkpoint metrics, latency, and throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from src.preprocessing.landmark_dataframe import effective_subset_landmark_count

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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

from models.base_model import BaseModel  # noqa: E402
from preprocessing.landmark_dataframe import (  # noqa: E402
    imputation_label,
    load_and_prepare_csv,
    subset_indices,
)
from representations.base_representation import BaseRepresentation  # noqa: E402
from training.lopo_dataset import LopoDataset  # noqa: E402


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "with"}:
        return True
    if normalized in {"false", "0", "no", "n", "without"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid Boolean value: {value}")


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError("device must be one of: auto, cuda, mps, cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def load_preprocessed_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    cache_path = (
        PROJECT_ROOT
        / "experiments"
        / f"{args.dataset}_speed_eval_cache"
        / "preprocessed"
        / f"{args.dataset}_{args.subset}_{imputation_label(args.imputation)}.csv"
    )

    if args.cache_preprocessed and cache_path.exists() and not args.force_preprocess:
        print(f"Using preprocessed CSV cache: {cache_path}")
        return pd.read_csv(cache_path)

    allowed_values = {"train", "val", "test"} if args.person_col == "split" else None
    dataframe = load_and_prepare_csv(
        args.data_csv,
        subset=args.subset,
        use_imputation=args.imputation,
        category_col=args.category_col,
        video_col=args.video_col,
        frame_col=args.frame_col,
        person_col=args.person_col,
        lowercase_person=args.person_col == "split",
        allowed_person_values=allowed_values,
    )

    if args.cache_preprocessed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(cache_path, index=False)
        print(f"Preprocessed CSV cache saved to {cache_path}")

    return dataframe


def build_model(
    model_name: str,
    num_classes: int,
) -> tuple[torch.nn.Module, tuple[int, int], object | None]:
    model_class = BaseModel.get_by_name(model_name)
    if model_class is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: "
            f"{BaseModel.available_names()}"
        )
    base_model = model_class(num_classes)
    return (
        base_model.get_model(),
        base_model.image_size,
        base_model.get_transforms(),
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format; expected a state dictionary.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)


def save_confusion_matrix(
    path: Path,
    labels: list[int],
    predictions: list[int],
    title: str,
) -> None:
    matrix = confusion_matrix(labels, predictions)
    figure_size = max(8, min(20, matrix.shape[0] * 0.35))
    plt.figure(figsize=(figure_size, figure_size))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def timing_statistics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "max", "mean", "std", "median")}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate classification metrics and inference speed for a checkpoint."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-csv", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--subset", required=True, choices=["all", "1st", "2nd", "laines", "arcanjo"]
    )
    parser.add_argument("--imputation", required=True, type=str_to_bool)
    parser.add_argument("--category-col", default="sign_id")
    parser.add_argument("--video-col", default="sequence_id")
    parser.add_argument("--frame-col", default="frame_id")
    parser.add_argument("--person-col", default="split")
    parser.add_argument("--eval-person", default="test")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--image-method", default="Skeleton-DML")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu", "mps", "auto"]
    )
    parser.add_argument("--cache-preprocessed", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--save-confusion-matrix", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters cannot be negative")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = (
            PROJECT_ROOT
            / "reports"
            / "speed"
            / f"{args.dataset}_{args.subset}_{imputation_label(args.imputation)}_{timestamp}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Checkpoint speed evaluation")
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Dataset:        {args.dataset}")
    print(f"CSV:            {args.data_csv}")
    print(f"Checkpoint:     {args.checkpoint_path}")
    print(f"Subset:         {args.subset} ({len(effective_subset_landmark_count(args.subset))} landmarks)")
    print(f"Imputation:     {imputation_label(args.imputation)}")
    print(f"Evaluation key: {args.person_col}={args.eval_person}")
    print(f"Output folder:  {args.output_dir}")
    print("=" * 72)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataframe = load_preprocessed_dataframe(args)
    people = dataframe["person"].astype(str)
    eval_value = str(args.eval_person)
    eval_dataframe = dataframe[people == eval_value]
    if eval_dataframe.empty:
        available = sorted(people.unique().tolist())
        raise ValueError(
            f"No samples found for {args.person_col}={args.eval_person}. "
            f"Available values: {available}"
        )

    full_class_count = int(dataframe["category"].nunique())
    eval_video_count = int(eval_dataframe["video_name"].nunique())
    print(f"\nEvaluation frames:  {len(eval_dataframe):,}")
    print(f"Evaluation videos:  {eval_video_count:,}")
    print(f"Full class count:   {full_class_count}")
    print(f"Evaluation classes: {eval_dataframe['category'].nunique()}")

    representation_class = BaseRepresentation.get_by_name(args.image_method)
    if representation_class is None:
        raise ValueError(
            f"Unknown representation '{args.image_method}'. Available representations: "
            f"{BaseRepresentation.available_names()}"
        )
    image_method = representation_class()

    model, image_size, custom_transform = build_model(args.model, full_class_count)
    device = resolve_device(args.device)
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__} | CUDA runtime: {torch.version.cuda}")

    load_checkpoint(model, args.checkpoint_path, device)
    model.to(device)
    model.eval()

    transform = custom_transform or transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    )
    evaluation_dataset = LopoDataset(
        dataframe,
        image_method,
        transform,
        augment=False,
        seed=args.seed,
        person_in=[eval_value],
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"\nEvaluation samples: {len(evaluation_dataset)}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Number of batches:  {len(evaluation_loader)}")

    if args.warmup_iters:
        print(f"\nWarm-up iterations: {args.warmup_iters}")
        dummy_batch = torch.randn(
            args.batch_size,
            3,
            image_size[0],
            image_size[1],
            device=device,
        )
        with torch.no_grad():
            for _ in range(args.warmup_iters):
                model(dummy_batch)
        synchronize(device)

    predictions: list[int] = []
    labels: list[int] = []
    forward_batch_ms: list[float] = []

    print("\nStarting evaluation...")
    end_to_end_start = time.perf_counter()
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(evaluation_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            synchronize(device)
            forward_start = time.perf_counter()
            output = model(data)
            synchronize(device)
            forward_end = time.perf_counter()

            predicted = output.argmax(dim=1)
            predictions.extend(predicted.cpu().numpy().astype(int).tolist())
            labels.extend(target.cpu().numpy().astype(int).tolist())
            forward_batch_ms.append((forward_end - forward_start) * 1000.0)

            if batch_index % 25 == 0:
                print(
                    f"Batch {batch_index:>5}/{len(evaluation_loader)} | "
                    f"forward={forward_batch_ms[-1]:.2f} ms"
                )

    synchronize(device)
    end_to_end_seconds = time.perf_counter() - end_to_end_start
    forward_seconds = sum(forward_batch_ms) / 1000.0
    sample_count = len(labels)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

    results = {
        "dataset": args.dataset,
        "protocol": "checkpoint_speed_evaluation",
        "data_csv": str(args.data_csv),
        "checkpoint_path": str(args.checkpoint_path),
        "subset": args.subset,
        "subset_landmarks": effective_subset_landmark_count(args.subset),
        "imputation": args.imputation,
        "imputation_label": imputation_label(args.imputation),
        "category_col": args.category_col,
        "video_col": args.video_col,
        "frame_col": args.frame_col,
        "person_col": args.person_col,
        "eval_person": args.eval_person,
        "model": args.model,
        "image_method": args.image_method,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "num_classes": full_class_count,
        "num_eval_samples": sample_count,
        "num_eval_videos": len(evaluation_dataset),
        "batch_size": args.batch_size,
        "warmup_iters": args.warmup_iters,
        "end_to_end_seconds": float(end_to_end_seconds),
        "end_to_end_ms_per_sample": float(
            end_to_end_seconds / max(sample_count, 1) * 1000.0
        ),
        "end_to_end_samples_per_second": float(
            sample_count / end_to_end_seconds
        )
        if end_to_end_seconds > 0
        else None,
        "model_forward_seconds": float(forward_seconds),
        "model_forward_ms_per_sample": float(
            forward_seconds / max(sample_count, 1) * 1000.0
        ),
        "model_forward_samples_per_second": float(sample_count / forward_seconds)
        if forward_seconds > 0
        else None,
        "forward_batch_timing_ms": timing_statistics(forward_batch_ms),
        "metrics": {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
        },
        "evaluated_at": datetime.now().isoformat(),
    }

    result_path = args.output_dir / "speed_eval_results.json"
    save_json(result_path, results)

    confusion_matrix_path: Path | None = None
    if args.save_confusion_matrix:
        confusion_matrix_path = args.output_dir / "confusion_matrix.png"
        save_confusion_matrix(
            confusion_matrix_path,
            labels,
            predictions,
            f"Confusion Matrix - {args.dataset.upper()} - {args.subset}",
        )

    print("\n" + "=" * 72)
    print("RESULTS")
    print(f"Evaluated samples:              {sample_count}")
    print(f"Evaluated videos:               {len(evaluation_dataset)}")
    print(f"End-to-end time:                {end_to_end_seconds:.4f} s")
    print(f"End-to-end time per sample:     {results['end_to_end_ms_per_sample']:.2f} ms")
    print(f"End-to-end samples per second:  {results['end_to_end_samples_per_second']:.2f}")
    print(f"Model forward time per sample:  {results['model_forward_ms_per_sample']:.2f} ms")
    print(f"Model forward samples/second:   {results['model_forward_samples_per_second']:.2f}")
    print("-" * 72)
    print(f"Accuracy:                       {accuracy:.4f}")
    print(f"Weighted precision:             {precision:.4f}")
    print(f"Weighted recall:                {recall:.4f}")
    print(f"Weighted F1-score:              {f1:.4f}")
    print("-" * 72)
    print(f"JSON saved to:                  {result_path}")
    if confusion_matrix_path is not None:
        print(f"Confusion matrix saved to:      {confusion_matrix_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
