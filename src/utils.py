import torch
from torch import nn


def calc_dim():
    modules = []
    in_channels = 3
    hidden_dims = [32, 64, 128, 256, 512]

    # Build Encoder
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
        )
        in_channels = h_dim
    encoder = nn.Sequential(*modules)
    print(encoder(torch.rand((32, 3, 64, 64))).shape)


if __name__ == '__main__':
    calc_dim()
