from __future__ import annotations

from extraction.base_extractor import BaseExtractor

__all__ = ["BaseExtractor", "MediaPipeExtractor", "DefaultVideoProcessor", "UFOPVideoProcessor"]


def __getattr__(name: str):
    """Load extractor-specific dependencies only when they are requested."""
    if name == "MediaPipeExtractor":
        from extraction.mediapipe_extractor import MediaPipeExtractor

        return MediaPipeExtractor
    if name in {"DefaultVideoProcessor", "UFOPVideoProcessor"}:
        from extraction.video_processor import DefaultVideoProcessor, UFOPVideoProcessor

        return {
            "DefaultVideoProcessor": DefaultVideoProcessor,
            "UFOPVideoProcessor": UFOPVideoProcessor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
