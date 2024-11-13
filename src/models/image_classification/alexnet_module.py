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

    def transfer_batch_to_device(self, batch: dict, device: torch.device, dataloader_idx: int) -> Any:
        result = {}
        for key,value in batch.items():
            if isinstance(value, dict):
                result[key] = {k: v.to(device) for k,v in value}
            else:
                result[key] = value.to(device)
        in_channels = batch['inputs'].shape[1]
        return result

    @classmethod
    def from_config(cls, config):
        return cls(config)

