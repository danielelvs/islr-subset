"""
Dataset loaders: video-based (MINDS, UFOP) and CSV-based (KSL, Include50).

Video-based datasets feed the extraction pipeline.
CSV-based datasets load pre-processed OpenPose CSVs directly for training.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────

class BaseVideoDataset(ABC):
    """Dataset whose raw data is video files to be processed by an extractor."""

    path_key: str = ""

    def __init__(self, base_path: str):
        self.dataset_path = os.path.join(base_path, self.path_key)
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset directory not found: {self.dataset_path}")

    @abstractmethod
    def prepare_data(self) -> list[tuple]:
        """Return list of (video_path, video_name, category, signaler, index)."""

    def get_processor(self, extractor):
        from extraction.video_processor import DefaultVideoProcessor
        return DefaultVideoProcessor(extractor)

    @staticmethod
    def create(dataset_name: str, base_path: str) -> "BaseVideoDataset":
        registry = {
            "minds": MINDSDataset,
            "ufop": UFOPDataset,
            "vlibrasil": VLibrasilDataset,
        }
        if dataset_name not in registry:
            raise ValueError(
                f"Unknown video dataset '{dataset_name}'. "
                f"Available: {list(registry.keys())}"
            )
        return registry[dataset_name](base_path)


class BaseCsvDataset(ABC):
    """Dataset that ships as a pre-processed CSV (no video extraction needed)."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a DataFrame ready for training."""

    @staticmethod
    def create(dataset_name: str, data_dir: str) -> "BaseCsvDataset":
        registry = {
            "ksl": KSLDataset,
            "include50": Include50Dataset,
        }
        if dataset_name not in registry:
            raise ValueError(
                f"Unknown CSV dataset '{dataset_name}'. "
                f"Available: {list(registry.keys())}"
            )
        return registry[dataset_name](data_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Video-based datasets
# ──────────────────────────────────────────────────────────────────────────────

class MINDSDataset(BaseVideoDataset):
    """
    Supports two layouts:

    Flat (all .mp4 in root — e.g. "01AcontecerSinalizador01-2.mp4"):
        data/raw/minds/01AcontecerSinalizador01-2.mp4

    Nested (original layout):
        data/raw/minds/Sinalizador01/Canon/video.mp4
    """
    path_key = "minds"

    def prepare_data(self) -> list[tuple]:
        import re
        videos = []

        entries = os.listdir(self.dataset_path)
        # Detect flat layout: has .mp4 files directly in root
        has_flat_mp4 = any(e.endswith(".mp4") for e in entries)

        if has_flat_mp4:
            # Flat: "01AcontecerSinalizador01-2.mp4"
            pattern = re.compile(r"^(\d{2})(.+?)Sinalizador(\d+)", re.IGNORECASE)
            for filename in entries:
                if not filename.endswith(".mp4"):
                    continue
                m = pattern.match(filename)
                if not m:
                    continue
                category     = m.group(1)          # "01"
                signaler_id  = int(m.group(3))     # 1
                video_path   = os.path.join(self.dataset_path, filename)
                videos.append((video_path, filename, category, signaler_id, signaler_id))
        else:
            # Nested: Sinalizador01/Canon/*.mp4
            for signaler in entries:
                signaler_dir = os.path.join(self.dataset_path, signaler, "Canon")
                if not os.path.isdir(signaler_dir):
                    continue
                signaler_id = int(signaler[-2:])
                for video in os.listdir(signaler_dir):
                    if not video.endswith(".mp4"):
                        continue
                    video_path = os.path.join(signaler_dir, video)
                    category   = video.split("Sinalizador")[0][2:]
                    videos.append((video_path, video, category, signaler_id, signaler_id))

        return videos


class UFOPDataset(BaseVideoDataset):
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
        return UFOPVideoProcessor(extractor, self.labels, self.frames_threshold)

    def _load_labels(self) -> dict:
        labels_path = os.path.join(self.dataset_path, "labels.txt")
        if not os.path.exists(labels_path):
            raise ValueError(f"Labels file not found: {labels_path}")
        labels = {}
        with open(labels_path) as f:
            for line in f:
                parts = line.strip().split(" ")
                labels[parts[0]] = parts[1:]
        return labels


class VLibrasilDataset(BaseVideoDataset):
    path_key = "v-librasil/videos UFPE (V-LIBRASIL)/data"

    def prepare_data(self) -> list[tuple]:
        videos = []
        for video in os.listdir(self.dataset_path):
            video_path = os.path.join(self.dataset_path, video)
            sign = video.split("_")[0]
            signaler = video[-5]
            videos.append((video_path, video, sign, signaler, signaler))
        return videos


# ──────────────────────────────────────────────────────────────────────────────
# CSV-based datasets
# ──────────────────────────────────────────────────────────────────────────────

class KSLDataset(BaseCsvDataset):
    """
    Korean Sign Language dataset.

    Expected CSV columns (MediaPipe format):
        sign, sign_id, interpreter, video_name, frame_id, hand_0_0_x, ...

    Column mapping:
        interpreter → person
        sign_id     → category  (already int)
        frame_id    → frame
        video_name  → video_name

    LOPO: 20 interpreters (0–19). Use sample_id groups like MINDS.
    """

    def __init__(self, data_dir: str):
       self.csv_path = os.path.join(data_dir, "ksl", "ksl_mediapipe.csv")

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"KSL CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Normalize column names to what the trainer expects
        df = df.rename(columns={
            "interpreter": "person",
            "frame_id":    "frame",
            "sign_id":     "category",
            "sequence_id": "video_name",
        })

        # category must be consecutive int starting at 0
        cats = sorted(df["category"].unique())
        df["category"] = df["category"].map({c: i for i, c in enumerate(cats)})
        # if df["category"].dtype == object or df["category"].nunique() != df["category"].max() + 1:
        #     cats = sorted(df["category"].unique())
        #     cat_map = {c: i for i, c in enumerate(cats)}
        #     df["category"] = df["category"].map(cat_map)

        return df


class Include50Dataset(BaseCsvDataset):
    """
    Include-50 dataset.

    Expected CSV columns (MediaPipe format):
        sign_id, category, sign, sign_key, sample_id, video_name,
        sequence_id, frame_id, hand_0_0_x, ...

    Since there is no interpreter/person column, we use sample_id as
    the person proxy for LOPO — each sample_id value represents one
    recording session across all signs.

    Column mapping:
        sample_id   → person
        sign_id     → category  (re-encoded to consecutive int)
        frame_id    → frame
        sequence_id → video_name  (unique per video)
    """

    def __init__(self, data_dir: str):
        self.csv_path = os.path.join(data_dir, "include50", "include50_mediapipe.csv")

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Include50 CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Use sequence_id as video_name (unique per video, no extension)
        if "sequence_id" in df.columns and "video_name" in df.columns:
            df["video_name"] = df["sequence_id"]

        # sample_id → person (LOPO proxy)
        df = df.rename(columns={
            "sample_id": "person",
            "frame_id":  "frame",
        })

        # Re-encode category to consecutive int
        # category column is a string like "19. House" — use sign_id if available
        if "sign_id" in df.columns and df["sign_id"].notna().any():
            df["category"] = df["sign_id"]
        # else keep category as-is and encode
        if df["category"].dtype == object:
            cats = sorted(df["category"].unique())
            df["category"] = df["category"].map({c: i for i, c in enumerate(cats)})

        # Ensure consecutive int
        cats = sorted(df["category"].unique())
        df["category"] = df["category"].map({c: i for i, c in enumerate(cats)})

        return df
