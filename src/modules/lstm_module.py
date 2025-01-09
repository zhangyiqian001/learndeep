import torch
from torch import nn
from torch.nn import LSTM

from modules.base_module import BaseRNNGenerateModule


# input_size: int
# hidden_size: int
# num_layers: int
# bias: bool
# batch_first: bool
# dropout: float
# bidirectional: bool
class LSTMGenerate(nn.Module):
    def __init__(
            self,
            hidden_size=20,
            num_layers=2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model = LSTM(
            input_size=1*28,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = self.model(x, (h0, c0))
        return x


class LSTMGenerateModule(BaseRNNGenerateModule):
    def __init__(self, model, loss, ):
        super().__init__()
        self.model = model
        self.loss = loss
        # self.transform = transform

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        batch_size, channels, height, width = batch[0].size()
        inputs = batch[0].permute(0, 2, 3, 1)  # 形状变为 (batch_size, sequence_length, input_size)
        inputs = inputs.view(batch_size, -1, channels)
        return {
            "inputs": inputs,
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
