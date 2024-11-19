from typing import Any
import torch
from lightning.pytorch import LightningModule
from torch import Tensor


class BaseModelModule(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, **kwargs) -> Tensor:
        return self.model(kwargs)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        inputs, target = batch['inputs'], batch['targets']
        output = self(inputs)
        loss = self.loss(output, target)
        acc = self.metrics(output.argmax(1), target)
        values = {"train_loss": loss, "train_acc": acc}
        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        inputs, target = batch['inputs'], batch['targets']
        output = self(inputs)
        loss = self.loss(output, target)
        acc = self.metrics(output.argmax(1), target)
        values = {"val_loss": loss, "val_acc": acc}
        self.log_dict(values, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx):
        inputs = batch['inputs']
        output = self(inputs)
        return output

    def transfer_batch_to_device(self, batch: dict, device: torch.device, dataloader_idx: int) -> Any:
        result = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                result[key] = {k: v.to(device) for k, v in value}
            else:
                result[key] = [v.to(device) for v in value]
        return result


class BaseTranslateModelModule(LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs) -> Tensor:
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        output = self(**batch)
        loss = self.loss(output.view(-1, output.shape[-1]), batch['targets'].view(-1))
        # acc = self.metrics(output, target)
        values = {"train_loss": loss}
        # values = {"train_loss": loss, "train_acc": acc}
        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        output = self(**batch)
        loss = self.loss(output.view(-1, output.shape[-1]), batch['targets'].view(-1))
        # acc = self.metrics(output, target)
        values = {"val_loss": loss}
        # values = {"val_loss": loss, "val_acc": acc}
        self.log_dict(values, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx):
        inputs = batch['inputs']
        output = self(inputs)
        return output

    def transfer_batch_to_device(self, batch: dict, device: torch.device, dataloader_idx: int) -> Any:
        result = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                result[key] = {k: v.to(device) for k, v in value}
            else:
                result[key] = value.to(device)
        return result