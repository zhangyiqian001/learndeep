from typing import Any

import torch
from torch import nn
from torchvision.models import AlexNet

from models.base_module import BaseModelModule
from utils.registry import registry


@registry.register_model("alexnet")
class AlexNetModule(BaseModelModule):
    def __init__(self, config):
        super().__init__(config)
        args = config.ARCH_CONFIG
        self.model = AlexNet(**args)
        self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    @classmethod
    def from_config(cls, config):
        return cls(config)

