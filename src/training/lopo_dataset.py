"""
LopoDataset — Leave-One-Person-Out dataset for skeleton-based ISLR.

Bugs fixed vs. original:
    1. mirror_flip: was flipping ~70% of the time due to inverted condition.
       Fixed to flip with probability `mirror_chance` (default 0.5 per paper).
    2. mirror_flip formula: was `-np.flip(df, axis=1)` which negates values and
       flips the temporal axis. Fixed to `1.0 - df` (proper x-coordinate reflection).
    3. state_dict deepcopy: trainer now uses copy.deepcopy (handled in trainer.py).
"""

import math
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────

def rotate_landmarks(x, y, z, angle_rad: float):
    pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    xy_rot = (pts[:, :2] - [cx, cy]) @ R.T + [cx, cy]
    return (
        xy_rot[:, 0].reshape(x.shape),
        xy_rot[:, 1].reshape(y.shape),
        z.copy(),
    )


def zoom_landmarks(arr: np.ndarray, factor: float) -> np.ndarray:
    return arr * factor


def translate_landmarks(arr: np.ndarray, delta: float) -> np.ndarray:
    return arr + delta


def mirror_flip_x(x: np.ndarray, chance: float) -> np.ndarray:
    """
    Reflect x coordinates horizontally with probability `chance`.
    FIX: was `np.random.random() > chance` (flips 1-chance of the time).
         Now `np.random.random() < chance` (flips `chance` of the time).
    FIX: was `-np.flip(df, axis=1)` (negates + reverses time).
         Now `1.0 - x` (proper horizontal reflection in [0,1] space).
    """
    if np.random.random() < chance:
        return 1.0 - x
    return x


def perform_augmentation(x, y, z, cfg: dict | None = None):
    """Apply stochastic augmentation to landmark matrices."""
    cfg = cfg or {}
    rotation   = np.random.normal(0, cfg.get("rotation_sigma",    12))
    zoom       = np.random.normal(0, cfg.get("zoom_sigma",        0.1)) + 1
    tx         = np.random.normal(0, cfg.get("translate_x_sigma", 0.06))
    ty         = np.random.normal(0, cfg.get("translate_y_sigma", 0.0))
    tz         = np.random.normal(0, cfg.get("translate_z_sigma", 0.0))
    flip_prob  = cfg.get("horizontal_flip_prob", 0.5)

    x, y, z = rotate_landmarks(x, y, z, math.radians(rotation))
    x = translate_landmarks(zoom_landmarks(x, zoom), tx)
    y = translate_landmarks(zoom_landmarks(y, zoom), ty)
    z = translate_landmarks(zoom_landmarks(z, zoom), tz)
    x = mirror_flip_x(x, flip_prob)
    return x, y, z


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

# Pose landmarks excluded from sign representation (below the hip)
_EXCLUDED_POSE_INDICES = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
_EXCLUDED_POSE_PREFIXES = tuple(f"pose_{i}" for i in _EXCLUDED_POSE_INDICES)


class LopoDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_method,
        transforms=None,
        augment: bool = True,
        augment_cfg: dict | None = None,
        person_in: list = (),
        person_out: list = (),
        seed: int | None = None,
    ):
        self.transforms = transforms
        self.image_method = image_method
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self.person_in = list(person_in)
        self.person_out = list(person_out)
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.signs = self._get_sign_columns(dataframe)
        self.categories = list(dataframe["category"].unique())
        self.df = self._prepare(dataframe)

    # ── Public ────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.df)

    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
    #     x, y, z = row["x"].copy(), row["y"].copy(), row["z"].copy()
    #     if self.augment:
    #         x, y, z = perform_augmentation(x, y, z, self.augment_cfg)
    #     image = self.image_method.transform(x, y, z)
    #     image = Image.fromarray(np.uint8(image * 255)).convert("RGB")
    #     if self.transforms:
    #         image = self.transforms(image)
    #     label = self.categories.index(row["category"])
    #     return image, torch.tensor(label, dtype=torch.int64)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video = row["video_name"]
        category = row["category"]

        # Carrega só o vídeo atual
        vdf = self._df[self._df["video_name"] == video].sort_values("frame") if "frame" in self._df.columns else self._df[self._df["video_name"] == video]
        vdf = vdf.drop(columns=["category", "video_name", "person"], errors="ignore")

        x = np.clip(self._get_axis(vdf, "_x").T.to_numpy(dtype='float32'), 0, 1)
        y = np.clip(self._get_axis(vdf, "_y").T.to_numpy(dtype='float32'), 0, 1)
        z = np.clip(self._get_axis(vdf, "_z").T.to_numpy(dtype='float32'), 0, 1)

        if self.augment:
            x, y, z = perform_augmentation(x, y, z, self.augment_cfg)

        image = self.image_method.transform(x, y, z)
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = Image.fromarray(np.uint8(image * 255)).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        label = self.categories.index(category)
        return image, torch.tensor(label, dtype=torch.int64)

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_sign_columns(self, df: pd.DataFrame) -> list[str]:
        cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
        # Remove below-hip pose landmarks (irrelevant for sign language)
        return [c for c in cols if not c.startswith(_EXCLUDED_POSE_PREFIXES)]

    def _get_axis(self, df: pd.DataFrame, axis: str) -> pd.DataFrame:
        return df[[c for c in self.signs if c.endswith(axis)]]

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # Guarda só metadados + índices, não converte pra numpy ainda
        keep = ["category", "video_name", "person"] + self.signs
        df = df[[c for c in keep if c in df.columns]]

        records = []
        for video in df["video_name"].unique():
            vdf = df[df["video_name"] == video]
            person = vdf.iloc[0]["person"]
            if self.person_in and person not in self.person_in:
                continue
            if self.person_out and person in self.person_out:
                continue
            records.append({
                "video_name": video,
                "category":   vdf.iloc[0]["category"],
                "person":     person,
            })

        # Guarda o df filtrado para acesso no __getitem__
        self._df = df
        return pd.DataFrame(records)

    # def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
    #     keep = ["category", "video_name", "person", "frame"] + self.signs
    #     df = df[[c for c in keep if c in df.columns]]
    #     records = []
    #     for video in df["video_name"].unique():
    #         vdf = df[df["video_name"] == video].sort_values("frame")
    #         person = vdf.iloc[0]["person"]
    #         if self.person_in and person not in self.person_in:
    #             continue
    #         if self.person_out and person in self.person_out:
    #             continue
    #         vdf = vdf.drop(columns=["category", "video_name", "frame"], errors="ignore")
    #         x = np.clip(self._get_axis(vdf, "_x").T.to_numpy(), 0, 1)
    #         y = np.clip(self._get_axis(vdf, "_y").T.to_numpy(), 0, 1)
    #         z = np.clip(self._get_axis(vdf, "_z").T.to_numpy(), 0, 1)
    #         records.append({
    #             "video_name": video,
    #             "category":   df[df["video_name"] == video].iloc[0]["category"],
    #             "x": x, "y": y, "z": z,
    #         })
    #     return pd.DataFrame(records)
