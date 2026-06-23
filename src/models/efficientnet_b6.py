from torch import nn
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from models.base_model import BaseModel


class EfficientNetB6Model(BaseModel):
    name = "efficientnet_b6"
    image_size = (224, 224)

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self._model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
        num_ftrs = self._model.classifier[1].in_features
        self._model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def get_model(self):
        return self._model

    def get_fc_layer(self):
        return self._model.classifier

    def get_transforms(self):
        return None
