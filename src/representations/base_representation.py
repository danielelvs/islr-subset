from abc import ABC, abstractmethod
import numpy as np


class BaseRepresentation(ABC):
    name: str

    @abstractmethod
    def transform(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Convert landmark matrices (n_landmarks × n_frames) to an image array."""

    @staticmethod
    def get_by_name(name: str) -> type | None:
        registry = {cls.name: cls for cls in BaseRepresentation.__subclasses__()}
        return registry.get(name)
