from __future__ import annotations
from abc import ABC, abstractmethod


class BaseModel(ABC):
    name: str
    image_size: tuple

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_fc_layer(self):
        pass

    @abstractmethod
    def get_transforms(self):
        pass

    @staticmethod
    def get_by_name(name: str) -> type | None:
        registry = {cls.name: cls for cls in BaseModel.__subclasses__()}
        return registry.get(name)
