import torch
import torchmetrics

from modules.base_module import BaseClassificationModule
from torchvision.models import SwinTransformer
from lightning.pytorch.profilers import PyTorchProfiler

class SwinTransformerModule(BaseClassificationModule):

    def __init__(self, model, loss, metrics):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metrics = metrics

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        return {
            "inputs": batch[0],
            "targets": batch[1],
        }

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        result = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                result[key] = {k: v.to(device) for k, v in value}
            else:
                result[key] = value.to(device)
        return result


swin_t = SwinTransformerModule(
    SwinTransformer(
        [4, 4],
        96,
        [2, 2, 6, 2],
        [3, 6, 12, 24],
        [7, 7]
    ),
    torch.nn.CrossEntropyLoss(),
    torchmetrics.Accuracy("multiclass", num_classes=2)
)
