from typing import Any

import torch

from datasets import BaseDataModule


class MnistDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        pass

    @classmethod
    def from_config(cls, config="default"):
        return cls(

        )