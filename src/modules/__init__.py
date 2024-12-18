from .alexnet_module import AlexNetModule1x3, AlexNetModule3x3
from .transformer_module import TransformerModule
from .kan_module import KANModule
from .vae_module import VAEModule
from .vqvae_module import VQVAEModule

__all__ = [
    "AlexNetModule1x3",
    "AlexNetModule3x3",
    "TransformerModule",
    "VAEModule",
    "VQVAEModule"
]

