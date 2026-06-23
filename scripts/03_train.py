#!/usr/bin/env python
"""
Step 3 — Train a model with LOPO (Leave-One-Person-Out) cross-validation.

Works for MediaPipe-filtered datasets (minds, ufop) and
pre-processed OpenPose CSV datasets (ksl, include50).

Usage examples:
    # MINDS — subset 2nd, resnet18, person 2 as test
    python scripts/03_train.py \\
        --config configs/minds_2nd_resnet18.yaml \\
        -vp 1 -tp 2

    # Manual (no YAML)
    python scripts/03_train.py \\
        -d minds -s 2nd -m resnet18 -im Skeleton-DML \\
        -lr 0.0001 -wd 0.0001 -r 63 -vp 1 -tp 2

    # KSL (OpenPose CSV, no subset needed)
    python scripts/03_train.py \\
        -d ksl -m resnet18 -im Skeleton-DML \\
        -lr 0.0001 -wd 0.0001 -r 15 \\
        -vp 0,1,2,3 -tp 0,1,2,3

    # Include50 (pre-split CSV: person 0=train, 1=val, 2=test)
    python scripts/03_train.py \\
        -d include50 -m resnet18 -im Skeleton-DML \\
        -lr 0.0001 -wd 0.0001 -r 15 -vp 1 -tp 2
"""

import argparse
import os
import sys
import re

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.base_dataset import BaseCsvDataset
from training.trainer import Trainer

CSV_DATASETS = {"ksl", "include50"}
VIDEO_DATASETS = {"minds", "ufop", "vlibras"}


def parse_people(value: str) -> list[int]:
    if not value or value == "-1":
        return []
    return [int(v) for v in value.split(",")]


def load_df(args) -> pd.DataFrame:
    dataset = args.dataset

    if dataset in CSV_DATASETS:
        loader = BaseCsvDataset.create(dataset, args.data_dir)
        return loader.load()

    # MediaPipe filtered CSV
    suffix = "" if not args.no_impute else "_no_imputation"
    subset = args.subset or "2nd"
    csv = os.path.join(args.data_dir, "processed", dataset, f"{dataset}_{subset}{suffix}.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(
            f"Processed CSV not found: {csv}\n"
            f"Run 02_filter_landmarks.py first, or check --subset / --data-dir."
        )
    df = pd.read_csv(csv)

    # MINDS: extract person from video_name if missing
    if dataset == "minds" and "person" not in df.columns:
        df["person"] = df["video_name"].apply(
            lambda v: int(re.findall(r".*Sinalizador(\d+)-.+\.mp4", str(v))[0])
        )
    return df


def main():
    parser = argparse.ArgumentParser(description="Train ISLR model")
    parser.add_argument("--config", help="YAML config file (overrides CLI args)")
    parser.add_argument("-d", "--dataset", help="Dataset name")
    parser.add_argument("-s", "--subset", default="2nd", help="Landmark subset (for MediaPipe datasets)")
    parser.add_argument("-m", "--model", default="resnet18")
    parser.add_argument("-im", "--image_method", default="Skeleton-DML")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("-r", "--ref", default="01", help="Experiment reference ID")
    parser.add_argument("-vp", "--validate_people", default="")
    parser.add_argument("-tp", "--test_people", required=False)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--no-impute", action="store_true")
    args = parser.parse_args()

    # ── Load YAML config (if provided) ───────────────────────────────────────
    cfg: dict = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides YAML for explicitly provided values
    dataset      = args.dataset         or cfg.get("dataset")
    subset       = args.subset          or cfg.get("subset", "2nd")
    model_name   = args.model           or cfg.get("model", "resnet18")
    image_method = args.image_method    or cfg.get("image_method", "Skeleton-DML")
    lr           = args.learning_rate   or cfg.get("training", {}).get("learning_rate", 1e-4)
    wd           = args.weight_decay    or cfg.get("training", {}).get("weight_decay", 1e-4)
    ref          = args.ref             or cfg.get("experiment_name", "01")
    epochs       = args.epochs          or cfg.get("training", {}).get("epochs", 30)
    seed         = args.seed            or cfg.get("training", {}).get("seed", 42)
    val_ppl      = parse_people(args.validate_people) or cfg.get("evaluation", {}).get("validate_people", [])
    test_ppl     = parse_people(args.test_people or "") or cfg.get("evaluation", {}).get("test_people", [])

    if not dataset:
        parser.error("--dataset is required (or set 'dataset' in YAML config)")
    if not test_ppl:
        parser.error("--test_people is required")

    args.dataset    = dataset
    args.subset     = subset
    args.no_impute  = args.no_impute or cfg.get("imputation") == "none"

    aug_cfg = cfg.get("augmentation", {})

    trainer_cfg = {
        "dataset_name":    dataset,
        "ref":             ref,
        "seed":            seed,
        "validate_people": val_ppl,
        "test_people":     test_ppl,
        "learning_rate":   lr,
        "weight_decay":    wd,
        "image_method":    image_method,
        "model":           model_name,
        "epochs":          epochs,
        "patience":        cfg.get("training", {}).get("patience", 5),
        "batch_size":      cfg.get("training", {}).get("batch_size", 64),
        "augment_cfg":     aug_cfg,
        "output_dir":      args.output_dir,
    }

    print(f"\n{'='*60}")
    print(f"Dataset:       {dataset} | Subset: {subset}")
    print(f"Model:         {model_name} | Representation: {image_method}")
    print(f"Validate:      {val_ppl} | Test: {test_ppl}")
    print(f"LR: {lr} | WD: {wd} | Epochs: {epochs} | Seed: {seed}")
    print(f"{'='*60}\n")

    df = load_df(args)
    Trainer(trainer_cfg).run(df)


if __name__ == "__main__":
    main()
