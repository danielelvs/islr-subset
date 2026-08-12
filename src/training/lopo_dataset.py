from __future__ import annotations

"""PyTorch dataset for person-independent ISLR experiments."""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def rotate_landmarks(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    angle_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate x/y coordinates around their centroid."""

    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    center_x, center_y = np.mean(points[:, 0]), np.mean(points[:, 1])
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rotated_xy = (
        (points[:, :2] - [center_x, center_y]) @ rotation.T
        + [center_x, center_y]
    )
    return (
        rotated_xy[:, 0].reshape(x.shape),
        rotated_xy[:, 1].reshape(y.shape),
        z.copy(),
    )


def zoom_landmarks(array: np.ndarray, factor: float) -> np.ndarray:
    return array * factor


def translate_landmarks(array: np.ndarray, delta: float) -> np.ndarray:
    return array + delta


def mirror_flip_x(
    x: np.ndarray,
    probability: float,
) -> np.ndarray:
    """Reflect normalized x coordinates with the requested probability."""

    if np.random.random() < probability:
        return 1.0 - x
    return x


def perform_augmentation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply stochastic geometric augmentation to landmark matrices."""

    config = config or {}
    rotation = np.random.normal(0, config.get("rotation_sigma", 12))
    zoom = np.random.normal(0, config.get("zoom_sigma", 0.1)) + 1
    translate_x = np.random.normal(0, config.get("translate_x_sigma", 0.06))
    translate_y = np.random.normal(0, config.get("translate_y_sigma", 0.0))
    translate_z = np.random.normal(0, config.get("translate_z_sigma", 0.0))
    flip_probability = config.get("horizontal_flip_prob", 0.5)

    x, y, z = rotate_landmarks(x, y, z, math.radians(rotation))
    x = translate_landmarks(zoom_landmarks(x, zoom), translate_x)
    y = translate_landmarks(zoom_landmarks(y, zoom), translate_y)
    z = translate_landmarks(zoom_landmarks(z, zoom), translate_z)
    x = mirror_flip_x(x, flip_probability)
    return x, y, z


class LopoDataset(Dataset):
    """Create one image sample per video.

    ``person_in`` keeps only the selected people/groups, while ``person_out``
    excludes them. The same mechanism is also used by INCLUDE-50, where the
    ``person`` column contains the fixed split name (``train``, ``val``, or
    ``test``).
    """

    REQUIRED_COLUMNS = {"category", "video_name", "person"}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_method,
        transforms=None,
        augment: bool = True,
        augment_cfg: dict | None = None,
        person_in: Sequence = (),
        person_out: Sequence = (),
        seed: int | None = None,
    ):
        missing = self.REQUIRED_COLUMNS - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"LopoDataset is missing required columns: {sorted(missing)}"
            )

        self.transforms = transforms
        self.image_method = image_method
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self.person_in = set(person_in)
        self.person_out = set(person_out)

        if seed is not None:
            np.random.seed(seed)

        self.sign_columns = self._get_sign_columns(dataframe)
        if not self.sign_columns:
            raise ValueError("No landmark coordinate columns were found.")

        self.categories = list(dataframe["category"].drop_duplicates())
        self.category_to_index = {
            category: index for index, category in enumerate(self.categories)
        }
        self._df, self._video_indices, self.samples = self._prepare(dataframe)

        if self.samples.empty:
            raise ValueError(
                "The dataset is empty after applying person_in/person_out filters."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples.iloc[index]
        video_name = sample["video_name"]
        category = sample["category"]

        video_df = self._df.loc[self._video_indices[video_name]]
        if "frame" in video_df.columns:
            video_df = video_df.sort_values("frame")
        video_df = video_df.drop(
            columns=["category", "video_name", "person", "frame"],
            errors="ignore",
        )

        x = np.clip(
            self._get_axis(video_df, "_x").T.to_numpy(dtype="float32"), 0, 1
        )
        y = np.clip(
            self._get_axis(video_df, "_y").T.to_numpy(dtype="float32"), 0, 1
        )
        z = np.clip(
            self._get_axis(video_df, "_z").T.to_numpy(dtype="float32"), 0, 1
        )

        if self.augment:
            x, y, z = perform_augmentation(x, y, z, self.augment_cfg)

        image = self.image_method.transform(x, y, z)
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0)
        image = Image.fromarray(np.uint8(image * 255)).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        label = self.category_to_index[category]
        return image, torch.tensor(label, dtype=torch.int64)

    @staticmethod
    def _get_sign_columns(df: pd.DataFrame) -> list[str]:
        """Return all coordinate columns without silently dropping pose points."""

        return [
            column
            for column in df.columns
            if column.endswith(("_x", "_y", "_z"))
        ]

    def _get_axis(self, df: pd.DataFrame, axis: str) -> pd.DataFrame:
        return df[
            [column for column in self.sign_columns if column.endswith(axis)]
        ]

    def _prepare(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[object, np.ndarray], pd.DataFrame]:
        keep_columns = [
            "category",
            "video_name",
            "person",
            "frame",
            *self.sign_columns,
        ]
        selected = df[
            [column for column in keep_columns if column in df.columns]
        ]
        if self.person_in:
            selected = selected[selected["person"].isin(self.person_in)]
        if self.person_out:
            selected = selected[~selected["person"].isin(self.person_out)]

        working_df = selected.copy().reset_index(drop=True)

        records: list[dict] = []
        video_indices: dict[object, np.ndarray] = {}

        for video_name, group in working_df.groupby("video_name", sort=False):
            person = group.iloc[0]["person"]
            categories = group["category"].drop_duplicates()
            if len(categories) != 1:
                raise ValueError(
                    f"Video '{video_name}' contains multiple categories: "
                    f"{categories.tolist()}"
                )

            video_indices[video_name] = group.index.to_numpy()
            records.append(
                {
                    "video_name": video_name,
                    "category": categories.iloc[0],
                    "person": person,
                }
            )

        return working_df, video_indices, pd.DataFrame(records)
