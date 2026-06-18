from .base_model import BaseModel
from torch import nn
import timm


class VitMediumModel(BaseModel):
    name = "vit_medium"
    model = None
    transforms = None
    image_size = (256, 256)

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.model = timm.create_model(
            'vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k',
            pretrained=True,
            # num_classes=num_classes,  # remove classifier nn.Linear
        )
        # self.model.head = nn.Sequential(
        #     nn.Linear(512, num_classes),
        #     # nn.Softmax(dim=1)
        # )

        self.model.head = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def get_model(self):
        return self.model

    def get_fc_layer(self):
        return self.model.head

    def get_transformers(self):
        return self.transforms
