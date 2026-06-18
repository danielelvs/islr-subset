from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    @abstractmethod
    def get_video_landmarks(self, video_path):
        """Extract landmarks from a video file."""
        pass

    @abstractmethod
    def extract_landmarks(self, results):
        """Extract landmarks from a single frame's results."""
        pass

    @staticmethod
    def create(extractor_name):
        if extractor_name == "mediapipe":
            from mediapipe_extractor import MediaPipeExtractor
            return MediaPipeExtractor()
        elif extractor_name == "openpose":
            from openpose_extractor import OpenPoseExtractor
            return OpenPoseExtractor()
        raise ValueError(f"Extractor '{extractor_name}' not recognized. Use 'mediapipe' or 'openpose'.")