#!/usr/bin/env python3
"""Train a single LOPO fold or a fixed INCLUDE-50 split.

Examples
--------
MINDS, signer 1 for validation and signer 2 for testing::

    python scripts/03_train.py \
      --dataset minds --subset 2nd \
      --validate-people 1 --test-people 2

INCLUDE-50 fixed split::

    python scripts/03_train.py \
      --dataset include50 \
      --validate-people val --test-people test
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.landmark_dataframe import load_and_prepare_csv
from training.trainer import Trainer



def parse_group_values(value: str | None) -> list[int | str]:
    if not value:
        return []
    parsed: list[int | str] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        parsed.append(int(stripped) if stripped.lstrip("-").isdigit() else stripped)
    return parsed


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def choose(cli_value, config_value, default):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def load_dataframe(
    *,
    dataset: str,
    subset: str,
    no_imputation: bool,
    data_dir: str,
) -> pd.DataFrame:
    if dataset == "ksl":
        return load_and_prepare_csv(
            Path(data_dir) / "interim" / "ksl" / "ksl_mediapipe.csv",
            subset=subset,
            use_imputation=not no_imputation,
        )

    if dataset == "include50":
        return load_and_prepare_csv(
            Path(data_dir)
            / "interim"
            / "include50"
            / "include50_mediapipe_with_split.csv",
            subset=subset,
            use_imputation=not no_imputation,
            person_col="split",
            category_col="sign_id",
            video_col="sequence_id",
            frame_col="frame_id",
            lowercase_person=True,
            allowed_person_values={"train", "val", "test"},
        )

    suffix = "_no_imputation" if no_imputation else ""
    csv_path = (
        Path(data_dir)
        / "processed"
        / dataset
        / f"{dataset}_{subset}{suffix}.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed CSV not found: {csv_path}. Run 02_filter_landmarks.py "
            "first or verify --subset and --data-dir."
        )

    dataframe = pd.read_csv(csv_path)
    if dataset == "minds" and "person" not in dataframe.columns:
        pattern = re.compile(r".*Sinalizador(\d+)-.+\.mp4", re.IGNORECASE)

        def extract_signer(video_name: str) -> int:
            match = pattern.match(str(video_name))
            if not match:
                raise ValueError(
                    f"Could not extract the signer id from video name: {video_name}"
                )
            return int(match.group(1))

        dataframe["person"] = dataframe["video_name"].map(extract_signer)
    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one ISLR experiment.")
    parser.add_argument("--config", help="Optional YAML configuration file")
    parser.add_argument("-d", "--dataset", default=None)
    parser.add_argument("-s", "--subset", default=None)
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-im", "--image-method", default=None)
    parser.add_argument("-lr", "--learning-rate", type=float, default=None)
    parser.add_argument("-wd", "--weight-decay", type=float, default=None)
    parser.add_argument("-r", "--ref", default=None)
    parser.add_argument("-vp", "--validate-people", default=None)
    parser.add_argument("-tp", "--test-people", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--no-imputation", action="store_true")
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})

    dataset = choose(args.dataset, config.get("dataset"), None)
    subset = choose(args.subset, config.get("subset"), "2nd")
    model_name = choose(args.model, config.get("model"), "resnet18")
    image_method = choose(
        args.image_method, config.get("image_method"), "Skeleton-DML"
    )
    learning_rate = choose(
        args.learning_rate, training_config.get("learning_rate"), 1e-4
    )
    weight_decay = choose(
        args.weight_decay, training_config.get("weight_decay"), 1e-4
    )
    reference = choose(args.ref, config.get("experiment_name"), "01")
    epochs = choose(args.epochs, training_config.get("epochs"), 30)
    batch_size = choose(args.batch_size, training_config.get("batch_size"), 64)
    patience = choose(args.patience, training_config.get("patience"), 5)
    num_workers = choose(args.num_workers, training_config.get("num_workers"), 0)
    seed = choose(args.seed, training_config.get("seed"), 42)
    device = choose(args.device, training_config.get("device"), "auto")

    validation_people = parse_group_values(args.validate_people)
    if not validation_people:
        validation_people = evaluation_config.get("validate_people", [])
    test_people = parse_group_values(args.test_people)
    if not test_people:
        test_people = evaluation_config.get("test_people", [])

    if not dataset:
        parser.error("--dataset is required or must be defined in the YAML file")
    if not test_people:
        parser.error("--test-people is required")

    no_imputation = args.no_imputation or config.get("imputation") == "none"
    dataframe = load_dataframe(
        dataset=dataset,
        subset=subset,
        no_imputation=no_imputation,
        data_dir=args.data_dir,
    )

    trainer_config = {
        "dataset_name": dataset,
        "ref": reference,
        "seed": seed,
        "validate_people": validation_people,
        "test_people": test_people,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "image_method": image_method,
        "model": model_name,
        "device": device,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "augment_cfg": config.get("augmentation", {}),
        "output_dir": args.output_dir,
        "save_model": args.save_model,
    }

    print("\n" + "=" * 64)
    print(f"Dataset:        {dataset}")
    print(f"Subset:         {subset}")
    print(f"Model:          {model_name}")
    print(f"Representation: {image_method}")
    print(f"Validation:     {validation_people}")
    print(f"Test:           {test_people}")
    print(f"Device:         {device}")
    print(f"LR / WD:        {learning_rate} / {weight_decay}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Workers:        {num_workers}")
    print(f"Seed:           {seed}")
    print("=" * 64 + "\n")

    Trainer(trainer_config).run(dataframe)


if __name__ == "__main__":
    main()
