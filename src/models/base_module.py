from typing import Tuple

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from torch import Tensor
from torchmetrics import Accuracy


def define_loss(config):
    pass


def define_optim(config):
    pass

def define_metric(config):
    test_acc = torchmetrics.Accuracy()

class BaseModelModule(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.model = None

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return self.model(inputs, target)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.cross_entropy(output, target)
        acc = Accuracy(task="multiclass", num_classes=4).to(self.device)
        values = {"loss": loss, "acc": acc(output.argmax(1), target.argmax(1))}  # add more items if needed
        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.cross_entropy(output, target)
        acc = Accuracy(task="multiclass", num_classes=4).to(self.device)
        values = {"loss": loss, "acc": acc(output.argmax(1), target.argmax(1))}  # add more items if needed
        self.log_dict(values, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.cross_entropy(output, target)
        values = {"loss": loss}  # add more items if needed
        self.log_dict(values, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # return torch.optim.SGD(self.model.parameters(), lr=0.1)
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

