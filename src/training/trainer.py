from __future__ import annotations
"""
Unified trainer for all datasets (MINDS, UFOP, KSL, Include50).

Bugs fixed vs. original:
    1. val_loss: was criterion(outputs, labels) from LAST batch only.
       Fixed to accumulate and average over all validation batches.
    2. best_model_weights: was model.state_dict() (shallow reference).
       Fixed to copy.deepcopy(model.state_dict()).
"""

import copy
import json
import os
import random
from datetime import datetime

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, config: dict):
        """
        config keys:
            dataset_name    str
            ref             str     (experiment identifier)
            seed            int
            validate_people list[int]
            test_people     list[int]
            learning_rate   float
            weight_decay    float
            image_method    str     (e.g. "Skeleton-DML")
            model           str     (e.g. "resnet18")
            epochs          int
            batch_size      int     (default 64)
            patience        int     (default 5)
            augment_cfg     dict    (augmentation hyperparams)
            output_dir      str     (default "../experiments")
        """
        self.cfg = config
        # self.device = torch.device("cpu")
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu")

    def run(self, df: pd.DataFrame) -> dict:
        cfg = self.cfg
        set_seed(cfg.get("seed", 42))

        # ── Representation ────────────────────────────────────────────────────
        rep_cls = BaseRepresentation.get_by_name(cfg["image_method"])
        if rep_cls is None:
            raise ValueError(f"Unknown representation: {cfg['image_method']}")
        image_method = rep_cls()

        # ── Model ─────────────────────────────────────────────────────────────
        num_classes = len(df["category"].unique())
        model_cls = BaseModel.get_by_name(cfg["model"])
        if model_cls is None:
            raise ValueError(f"Unknown model: {cfg['model']}")
        base_model = model_cls(num_classes)
        model = base_model.get_model().to(self.device)

        # ── Transforms ───────────────────────────────────────────────────────
        custom_tf = base_model.get_transforms()
        default_tf = transforms.Compose([
            transforms.Resize(base_model.image_size),
            transforms.ToTensor(),
        ])
        tf = custom_tf if custom_tf is not None else default_tf

        # ── Datasets ──────────────────────────────────────────────────────────
        val_people  = cfg.get("validate_people", [])
        test_people = cfg["test_people"]
        aug_cfg     = cfg.get("augment_cfg", {})
        seed        = cfg.get("seed", 42)
        bs          = cfg.get("batch_size", 64)

        train_ds = LopoDataset(
            df, image_method, tf, augment=True, augment_cfg=aug_cfg,
            person_out=val_people + test_people, seed=seed,
        )
        test_ds = LopoDataset(
            df, image_method, tf, augment=False,
            person_in=test_people, seed=seed,
        )
        val_ds = (
            LopoDataset(df, image_method, tf, augment=False,
                        person_in=val_people, seed=seed)
            if val_people else None
        )

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False) if val_ds else None

        # ── Optimiser ─────────────────────────────────────────────────────────
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )

        # ── Training loop ─────────────────────────────────────────────────────
        epochs  = cfg.get("epochs", 30)
        patience = cfg.get("patience", 5)

        history = {"loss": [], "accuracy": [], "val_accuracy": []}
        best_val_loss = float("inf")
        best_val_acc  = 0.0
        # FIX: deepcopy so the reference doesn't follow subsequent updates
        best_weights  = copy.deepcopy(model.state_dict())
        counter = 0
        last_epoch = 0

        for epoch in tqdm(range(epochs), desc="Epochs"):
            last_epoch = epoch
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                if len(inputs) <= 1:
                    continue
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            train_acc  = correct / total
            history["loss"].append(float(epoch_loss))
            history["accuracy"].append(float(train_acc))

            # ── Validation ────────────────────────────────────────────────────
            if val_loader:
                model.eval()
                val_correct, val_total = 0, 0
                # FIX: accumulate val_loss over ALL batches (not just last)
                val_loss_sum, val_batches = 0.0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        val_loss_sum += criterion(outputs, labels).item()
                        val_batches  += 1
                        preds = outputs.argmax(dim=1)
                        val_correct += (preds == labels).sum().item()
                        val_total   += labels.size(0)
                val_acc  = val_correct / val_total
                val_loss = val_loss_sum / val_batches
                history["val_accuracy"].append(float(val_acc))
            else:
                val_acc  = train_acc
                val_loss = epoch_loss

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"loss={epoch_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"{datetime.now()}"
            )

        # ── Evaluate on test set ──────────────────────────────────────────────
        model.load_state_dict(best_weights)
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                preds = model(inputs).argmax(dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"\nTest — acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

        result = {
            "dataset_name":      cfg["dataset_name"],
            "ref":               cfg["ref"],
            "seed":              cfg.get("seed", 42),
            "epochs":            epochs,
            "last_epoch":        last_epoch,
            "image_method":      cfg["image_method"],
            "model":             cfg["model"],
            "validate_people":   val_people,
            "test_people":       test_people,
            "optimizer":         {"lr": cfg["learning_rate"], "weight_decay": cfg["weight_decay"]},
            "history":           history,
            "true_labels":       y_true,
            "predicted_labels":  y_pred,
            "test_accuracy":     acc,
            "test_precision":    prec,
            "test_recall":       rec,
            "test_f1":           f1,
        }

        # ── Save ──────────────────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(
            cfg.get("output_dir", "experiments"),
            cfg["ref"],
            cfg["dataset_name"],
        )
        os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

        json_path  = os.path.join(out_dir, f"{ts}.json")
        model_path = os.path.join(out_dir, "models", f"{ts}.pth")

        with open(json_path, "w") as f:
            json.dump(result, f, default=int)

        if cfg.get("save_model", True):
            torch.save(model.state_dict(), model_path)
        # torch.save(model.state_dict(), model_path)

        print(f"Results → {json_path}")
        return result
