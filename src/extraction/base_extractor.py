from __future__ import annotations
from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    @abstractmethod
    def get_video_landmarks(self, video_path):
        """Extract landmarks from a video file (generator)."""
        pass

    @abstractmethod
    def extract_landmarks(self, results):
        """Extract landmarks from a single frame's results."""
        pass

    @staticmethod
    def create(extractor_name: str):
        if extractor_name == "mediapipe":
            from extraction.mediapipe_extractor import MediaPipeExtractor
            return MediaPipeExtractor()
        raise ValueError(
            f"Extractor '{extractor_name}' not recognized. The supported extractor is 'mediapipe'. "
            "The OpenPose implementation is retained as legacy reference code only."
        )
