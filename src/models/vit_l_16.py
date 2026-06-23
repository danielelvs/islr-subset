from torch import nn
from torchvision.models import vit_l_16, ViT_L_16_Weights
from models.base_model import BaseModel


class VitL16Model(BaseModel):
    name = "vit_l_16"
    image_size = (512, 512)

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self._model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        num_ftrs = self._model.heads.head.in_features
        self._model.heads = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def get_model(self):
        return self._model

    def get_fc_layer(self):
        return self._model.heads

    def get_transforms(self):
        return None
