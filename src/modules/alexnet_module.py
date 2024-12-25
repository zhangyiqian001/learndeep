from typing import Any

import torch
from torch import nn

from modules.base_module import BaseClassificationModule


class AlexNetModuleBase(BaseClassificationModule):
    def __init__(
            self,
            model,
            loss,
            metrics
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metrics = metrics

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return {
            "inputs": batch[0],
            "targets": batch[1],
        }

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int) -> Any:
        result = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                result[key] = {k: v.to(device) for k, v in value}
            else:
                result[key] = value.to(device)
        return result


class AlexNetModule1x3(AlexNetModuleBase):
    def __init__(
            self,
            model,
            loss,
            metrics
    ):
        super().__init__(
            model,
            loss,
            metrics
        )
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)


class AlexNetModule3x3(AlexNetModuleBase):
    def __init__(
            self,
            model,
            loss,
            metrics
    ):
        super().__init__(
            model,
            loss,
            metrics
        )
        self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
