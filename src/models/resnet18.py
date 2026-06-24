from __future__ import annotations
from torch import nn
from torchvision.models import resnet18
from models.base_model import BaseModel


class Resnet18Model(BaseModel):
    name = "resnet18"
    image_size = (224, 224)

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self._model = resnet18(pretrained=True)
        num_ftrs = self._model.fc.in_features
        self._model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def get_model(self):
        return self._model

    def get_fc_layer(self):
        return self._model.fc

    def get_transforms(self):
        return None
