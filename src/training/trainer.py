from __future__ import annotations

"""Unified training loop for MINDS, UFOP, KSL, and INCLUDE-50."""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.base_model import BaseModel
from representations.base_representation import BaseRepresentation
from training.lopo_dataset import LopoDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
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


def state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone model weights on CPU to avoid duplicating them in GPU memory."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


class Trainer:
    """Train one fold or one fixed train/validation/test split."""

    def __init__(self, config: dict):
        self.cfg = config
        self.device = resolve_device(config.get("device", "auto"))

    def run(self, df: pd.DataFrame) -> dict:
        config = self.cfg
        # seed = int(config.get("seed", 42))
        seed = int(config.get("seed", 1638102311))
        set_seed(seed)

        representation_class = BaseRepresentation.get_by_name(
            config["image_method"]
        )
        if representation_class is None:
            raise ValueError(
                f"Unknown representation: {config['image_method']}"
            )
        image_method = representation_class()

        num_classes = int(df["category"].nunique())
        model_class = BaseModel.get_by_name(config["model"])
        if model_class is None:
            raise ValueError(f"Unknown model: {config['model']}")
        base_model = model_class(num_classes)
        model = base_model.get_model().to(self.device)

        custom_transform = base_model.get_transforms()
        default_transform = transforms.Compose(
            [
                transforms.Resize(base_model.image_size),
                transforms.ToTensor(),
            ]
        )
        transform = custom_transform or default_transform

        validation_people = config.get("validate_people", [])
        test_people = config["test_people"]
        augmentation_config = config.get("augment_cfg", {})
        batch_size = int(config.get("batch_size", 64))
        num_workers = int(config.get("num_workers", 0))
        if batch_size < 2:
            raise ValueError(
                "Training batch_size must be at least 2 because the classifier "
                "head uses BatchNorm1d."
            )

        train_dataset = LopoDataset(
            df,
            image_method,
            transform,
            augment=True,
            augment_cfg=augmentation_config,
            person_out=validation_people + test_people,
            seed=seed,
        )
        test_dataset = LopoDataset(
            df,
            image_method,
            transform,
            augment=False,
            person_in=test_people,
            seed=seed,
        )
        validation_dataset = (
            LopoDataset(
                df,
                image_method,
                transform,
                augment=False,
                person_in=validation_people,
                seed=seed,
            )
            if validation_people
            else None
        )

        if len(train_dataset) < 2:
            raise ValueError(
                "At least two training videos are required because the classifier "
                "head uses BatchNorm1d."
            )

        # BatchNorm1d cannot train on a batch containing a single sample.
        drop_last = len(train_dataset) > 1 and len(train_dataset) % batch_size == 1
        loader_options = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            **loader_options,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_options,
        )
        validation_loader = (
            DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_options,
            )
            if validation_dataset is not None
            else None
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )

        epochs = int(config.get("epochs", 30))
        patience = int(config.get("patience", 5))
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }

        best_validation_loss = float("inf")
        best_validation_accuracy = float("-inf")
        max_validation_accuracy = float("-inf")
        best_epoch = 0
        best_weights = state_dict_to_cpu(model)
        epochs_without_improvement = 0
        last_epoch = 0

        for epoch_index in tqdm(range(epochs), desc="Epochs"):
            current_epoch = epoch_index + 1
            last_epoch = current_epoch
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in tqdm(
                train_loader,
                desc=f"Epoch {current_epoch}",
                leave=False,
            ):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                train_total += labels.size(0)

            if train_total == 0:
                raise RuntimeError("No training samples were processed.")

            train_loss = train_loss_sum / train_total
            train_accuracy = train_correct / train_total
            history["train_loss"].append(float(train_loss))
            history["train_accuracy"].append(float(train_accuracy))

            if validation_loader is not None:
                model.eval()
                validation_correct = 0
                validation_total = 0

                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs = inputs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                        outputs = model(inputs)

                        validation_correct += (
                            outputs.argmax(dim=1) == labels
                        ).sum().item()
                        validation_total += labels.size(0)

                if validation_total == 0:
                    raise RuntimeError("No validation samples were processed.")

                validation_accuracy = validation_correct / validation_total

                # Same validation-loss calculation used by Alves:
                # loss from the last validation batch.
                validation_loss = criterion(outputs, labels).item()

            else:
                validation_loss = train_loss
                validation_accuracy = train_accuracy

            # if validation_loader is not None:
            #     model.eval()
            #     validation_loss_sum = 0.0
            #     validation_correct = 0
            #     validation_total = 0
            #     with torch.no_grad():
            #         for inputs, labels in validation_loader:
            #             inputs = inputs.to(self.device, non_blocking=True)
            #             labels = labels.to(self.device, non_blocking=True)
            #             outputs = model(inputs)
            #             batch_loss = criterion(outputs, labels)
            #             validation_loss_sum += batch_loss.item() * inputs.size(0)
            #             validation_correct += (
            #                 outputs.argmax(dim=1) == labels
            #             ).sum().item()
            #             validation_total += labels.size(0)

            #     if validation_total == 0:
            #         raise RuntimeError("No validation samples were processed.")
            #     validation_loss = validation_loss_sum / validation_total
            #     validation_accuracy = validation_correct / validation_total
            # else:
            #     validation_loss = train_loss
            #     validation_accuracy = train_accuracy

            history["validation_loss"].append(float(validation_loss))
            history["validation_accuracy"].append(float(validation_accuracy))

            max_validation_accuracy = max(
                max_validation_accuracy,
                validation_accuracy,
            )

            # Save the checkpoint with the highest validation accuracy.
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = current_epoch
                best_weights = state_dict_to_cpu(model)

            # Early stopping remains controlled by validation loss.
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {current_epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} | "
                f"val_loss={validation_loss:.4f} "
                f"val_acc={validation_accuracy:.4f}"
            )

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {current_epoch}.")
                break

        # Training is finished. Load the checkpoint with the highest
        # validation accuracy before evaluating on the test set.
        model.load_state_dict(best_weights)
        model.eval()

        true_labels: list[int] = []
        predicted_labels: list[int] = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device, non_blocking=True)

                predictions = model(inputs).argmax(dim=1).cpu()

                batch_true = labels.numpy().astype(int).tolist()
                batch_pred = predictions.numpy().astype(int).tolist()

                if batch_true:
                    min_label = min(batch_true)
                    max_label = max(batch_true)

                    if min_label < 0 or max_label >= num_classes:
                        raise RuntimeError(
                            f"Invalid test labels before metric computation: "
                            f"min={min_label}, max={max_label}, "
                            f"num_classes={num_classes}"
                        )

                true_labels.extend(batch_true)
                predicted_labels.extend(batch_pred)

        if not true_labels:
            raise RuntimeError("No test samples were processed.")

        accuracy = accuracy_score(true_labels, predicted_labels)

        precision = precision_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )

        print(
            f"\nTest metrics | accuracy={accuracy:.4f} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f}"
        )

        result = {
            "dataset_name": config["dataset_name"],
            "ref": config["ref"],
            "seed": seed,
            "epochs": epochs,
            "last_epoch": last_epoch,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_validation_loss),
            "best_val_accuracy": float(best_validation_accuracy),
            "max_val_accuracy": float(max_validation_accuracy),
            "image_method": config["image_method"],
            "model": config["model"],
            "device": str(self.device),
            "validate_people": validation_people,
            "test_people": test_people,
            "optimizer": {
                "lr": config["learning_rate"],
                "weight_decay": config["weight_decay"],
            },
            "history": history,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "test_accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
        }

        output_dir = Path(config.get("output_dir", "experiments"))
        run_dir = output_dir / config["ref"] / config["dataset_name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_path = run_dir / f"{timestamp}.json"

        model_path: Path | None = None
        if config.get("save_model", True):
            configured_path = config.get("model_output_path")
            model_path = (
                Path(configured_path)
                if configured_path
                else run_dir / "models" / f"{timestamp}.pth"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            result["model_path"] = str(model_path)

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)

        print(f"Results saved to {json_path}")
        if model_path is not None:
            print(f"Model checkpoint saved to {model_path}")
        return result
