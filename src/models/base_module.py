from typing import Tuple

import torch
from lightning.pytorch import LightningModule
from torch import Tensor


class BaseModelModule(LightningModule):

    def __init__(self, config):
        super().__init__()

        self.model = None

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return self.model(inputs, target)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

