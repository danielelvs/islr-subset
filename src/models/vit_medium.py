from __future__ import annotations
import timm
from torch import nn
from models.base_model import BaseModel


class VitMediumModel(BaseModel):
    name = "vit_medium"
    image_size = (256, 256)

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self._model = timm.create_model(
            "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k",
            pretrained=True,
        )
        self._model.head = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        data_config = timm.data.resolve_model_data_config(self._model)
        self._transforms = timm.data.create_transform(**data_config, is_training=False)

    def get_model(self):
        return self._model

    def get_fc_layer(self):
        return self._model.head

    def get_transforms(self):
        return self._transforms
