from __future__ import annotations

"""Base class and lazy registry for image classification models."""

import importlib
from abc import ABC, abstractmethod

_MODEL_MODULES = {
    "resnet18": "models.resnet18",
    "resnet50": "models.resnet50",
    "efficientnet_b6": "models.efficientnet_b6",
    "vit_l_16": "models.vit_l_16",
    "vit_medium": "models.vit_medium",
    "mobilenet_v4_hybrid_medium": "models.mobilenet_v4",
}


class BaseModel(ABC):
    name: str
    image_size: tuple[int, int]

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    @abstractmethod
    def get_model(self):
        """Return the underlying PyTorch module."""

    @abstractmethod
    def get_fc_layer(self):
        """Return the model classification head."""

    @abstractmethod
    def get_transforms(self):
        """Return model-specific transforms or ``None``."""

    @staticmethod
    def get_by_name(name: str) -> type | None:
        module_name = _MODEL_MODULES.get(name)
        if module_name:
            importlib.import_module(module_name)
        registry = {subclass.name: subclass for subclass in BaseModel.__subclasses__()}
        return registry.get(name)

    @staticmethod
    def available_names() -> list[str]:
        return sorted(_MODEL_MODULES)
