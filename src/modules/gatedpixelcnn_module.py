import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base_module import BaseGenerate2Module
from modules.common_module import GatedMaskedConv2d


class GatedPixelCNN(nn.Module):
    def __init__(
            self,
            input_dim=256,
            dim=64,
            n_layers=15,
            n_classes=10
    ):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )
        self.input_dim = input_dim

    def forward(self, x, label):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, H)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x


class GatedPixelCNNModule(BaseGenerate2Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        return {
            "inputs": batch[0][:, 0].long(),
            "targets": batch[1],
        }
