from .alexnet_module import AlexNetModule1x3, AlexNetModule3x3
from .transformer_module import TransformerModule
from .kan_module import KANModule
from .vae_module import VAEModule
from .vqvae_module import VQVAEModule
from .pixelcnn_module import PixelCNNModule
from .gatedpixelcnn_module import GatedPixelCNNModule

__all__ = [
    "AlexNetModule1x3",
    "AlexNetModule3x3",
    "TransformerModule",
    "VAEModule",
    "VQVAEModule",
    "PixelCNNModule",
    "GatedPixelCNNModule",
]

