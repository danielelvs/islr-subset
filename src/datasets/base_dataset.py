from __future__ import annotations

"""Dataset loaders for video-based and CSV-based ISLR datasets."""

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseVideoDataset(ABC):
    """Base class for datasets that require landmark extraction from videos."""

    path_key: str = ""

    def __init__(self, base_path: str):
        self.dataset_path = os.path.join(base_path, self.path_key)
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset directory not found: {self.dataset_path}")

    @abstractmethod
    def prepare_data(self) -> list[tuple]:
        """Return ``(path, name, category, signer, index)`` tuples."""

    def get_processor(self, extractor):
        from extraction.video_processor import DefaultVideoProcessor

        return DefaultVideoProcessor(extractor)

    @staticmethod
    def create(dataset_name: str, base_path: str) -> "BaseVideoDataset":
        registry = {
            "minds": MINDSDataset,
            "ufop": UFOPDataset,
            "vlibrasil": VLibrasilDataset,
            "vlibras": VLibrasilDataset,
        }
        if dataset_name not in registry:
            raise ValueError(
                f"Unknown video dataset '{dataset_name}'. "
                f"Available datasets: {list(registry)}"
            )
        return registry[dataset_name](base_path)


class BaseCsvDataset(ABC):
    """Base class for datasets already represented as landmark CSV files."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a DataFrame normalized for the training pipeline."""

    @staticmethod
    def create(dataset_name: str, data_dir: str) -> "BaseCsvDataset":
        registry = {
            "ksl": KSLDataset,
            "include50": Include50Dataset,
        }
        if dataset_name not in registry:
            raise ValueError(
                f"Unknown CSV dataset '{dataset_name}'. "
                f"Available datasets: {list(registry)}"
            )
        return registry[dataset_name](data_dir)


class MINDSDataset(BaseVideoDataset):
    """MINDS-Libras video dataset loader supporting flat and nested layouts."""

    path_key = "minds"

    def prepare_data(self) -> list[tuple]:
        videos: list[tuple] = []
        entries = os.listdir(self.dataset_path)
        has_flat_mp4 = any(entry.lower().endswith(".mp4") for entry in entries)

        if has_flat_mp4:
            pattern = re.compile(
                r"^(\d{2})(.+?)Sinalizador(\d+)", re.IGNORECASE
            )
            for filename in entries:
                if not filename.lower().endswith(".mp4"):
                    continue
                match = pattern.match(filename)
                if not match:
                    continue
                category = match.group(1)
                signer_id = int(match.group(3))
                video_path = os.path.join(self.dataset_path, filename)
                videos.append(
                    (video_path, filename, category, signer_id, signer_id)
                )
        else:
            for signer_folder in entries:
                signer_dir = os.path.join(
                    self.dataset_path, signer_folder, "Canon"
                )
                if not os.path.isdir(signer_dir):
                    continue
                signer_id = int(signer_folder[-2:])
                for filename in os.listdir(signer_dir):
                    if not filename.lower().endswith(".mp4"):
                        continue
                    video_path = os.path.join(signer_dir, filename)
                    category = filename.split("Sinalizador")[0][2:]
                    videos.append(
                        (video_path, filename, category, signer_id, signer_id)
                    )

        return videos


class UFOPDataset(BaseVideoDataset):
    """LIBRAS-UFOP loader using ``labels.txt`` for temporal segmentation."""

    path_key = "ufop"

    def __init__(self, base_path: str):
        super().__init__(base_path)
        self.labels = self._load_labels()
        self.frames_threshold = 15

    def prepare_data(self) -> list[tuple]:
        videos = []
        for folder in os.listdir(self.dataset_path):
            if not folder.startswith("p"):
                continue
            subject_id = folder.split("_")[0][1:]
            video_path = os.path.join(self.dataset_path, folder, "Color.avi")
            videos.append((video_path, folder, -1, subject_id, subject_id))
        return videos

    def get_processor(self, extractor):
        from extraction.video_processor import UFOPVideoProcessor

        return UFOPVideoProcessor(
            extractor, self.labels, self.frames_threshold
        )

    def _load_labels(self) -> dict:
        labels_path = os.path.join(self.dataset_path, "labels.txt")
        if not os.path.exists(labels_path):
            raise ValueError(f"Labels file not found: {labels_path}")

        labels: dict[str, list[str]] = {}
        with open(labels_path, encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    labels[parts[0]] = parts[1:]
        return labels


class VLibrasilDataset(BaseVideoDataset):
    """V-LIBRASIL video dataset loader."""

    path_key = "v-librasil/videos UFPE (V-LIBRASIL)/data"

    def prepare_data(self) -> list[tuple]:
        videos = []
        for filename in os.listdir(self.dataset_path):
            video_path = os.path.join(self.dataset_path, filename)
            sign = filename.split("_")[0]
            signer = filename[-5]
            videos.append((video_path, filename, sign, signer, signer))
        return videos


class KSLDataset(BaseCsvDataset):
    """Korean Sign Language landmark CSV loader."""

    def __init__(self, data_dir: str):
        self.csv_path = Path(data_dir) / "ksl" / "ksl_mediapipe.csv"

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"KSL CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        df = df.rename(
            columns={
                "interpreter": "person",
                "frame_id": "frame",
                "sign_id": "category",
                "sequence_id": "video_name",
            }
        )

        required = {"person", "category", "video_name", "frame"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"KSL CSV is missing columns: {sorted(missing)}")

        categories = sorted(df["category"].dropna().unique())
        df["category"] = df["category"].map(
            {category: index for index, category in enumerate(categories)}
        )
        return df


class Include50Dataset(BaseCsvDataset):
    """INCLUDE-50 loader using a fixed ``train``/``val``/``test`` split.

    The normalized ``person`` column stores the split name so the existing
    training pipeline can separate the three partitions without pretending
    that ``sample_id`` is a signer identifier.
    """

    def __init__(self, data_dir: str):
        self.csv_path = (
            Path(data_dir)
            / "include50"
            / "include50_mediapipe_with_split.csv"
        )

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"INCLUDE-50 CSV not found: {self.csv_path}"
            )

        df = pd.read_csv(self.csv_path)
        required = {"sign_id", "sequence_id", "frame_id", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"INCLUDE-50 CSV is missing columns: {sorted(missing)}"
            )

        df["category"] = df["sign_id"]
        df["video_name"] = df["sequence_id"].astype(str)
        df["frame"] = df["frame_id"].astype(int)
        df["person"] = df["split"].astype(str).str.strip().str.lower()

        expected_splits = {"train", "val", "test"}
        found_splits = set(df["person"].dropna().unique())
        if found_splits != expected_splits:
            raise ValueError(
                f"INCLUDE-50 split values must be {sorted(expected_splits)}; "
                f"found {sorted(found_splits)}"
            )

        categories = sorted(df["category"].dropna().unique())
        df["category"] = df["category"].map(
            {category: index for index, category in enumerate(categories)}
        )
        return df
