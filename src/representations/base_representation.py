from __future__ import annotations

"""Base class and lazy registry for skeleton image representations."""

import importlib
from abc import ABC, abstractmethod

import numpy as np

_REPRESENTATION_MODULES = {
    "Skeleton-DML": "representations.skeleton_dml",
    "Skeleton-Magnitude": "representations.skeleton_magnitude",
    "SL-DML": "representations.sl_dml",
}


class BaseRepresentation(ABC):
    name: str

    @abstractmethod
    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """Convert landmark matrices into an image-like NumPy array."""

    @staticmethod
    def get_by_name(name: str) -> type | None:
        module_name = _REPRESENTATION_MODULES.get(name)
        if module_name:
            importlib.import_module(module_name)
        registry = {
            subclass.name: subclass
            for subclass in BaseRepresentation.__subclasses__()
        }
        return registry.get(name)

    @staticmethod
    def available_names() -> list[str]:
        return sorted(_REPRESENTATION_MODULES)
