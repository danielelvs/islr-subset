import timm
from torch import nn
from models.base_model import BaseModel


class MobilenetV4HybridMediumModel(BaseModel):
    name = "mobilenet_v4_hybrid_medium"
    image_size = (256, 256)

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self._model = timm.create_model(
            "mobilenetv4_hybrid_medium.e500_r224_in1k",
            pretrained=True,
        )
        self._model.classifier = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        data_config = timm.data.resolve_model_data_config(self._model)
        self._transforms = timm.data.create_transform(**data_config, is_training=False)

    def get_model(self):
        return self._model

    def get_fc_layer(self):
        return self._model.classifier

    def get_transforms(self):
        return self._transforms
