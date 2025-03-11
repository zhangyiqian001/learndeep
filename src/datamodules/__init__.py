from .torchtext_datamodule import TorchTextDataModule
from .torchvision_datamodule import TorchVisionDataModule, CelebADataModule
from .huggingface_datamodule import HuggingfaceDataModule, HuggingfaceTranslateDataModule
from .vctk_datamodule import VCTKDataModule
from .carvana_datamodule import CarvanaDataModule

__all__ = [
    "TorchTextDataModule",
    "TorchVisionDataModule",
    "CelebADataModule",
    "HuggingfaceDataModule",
    "HuggingfaceTranslateDataModule",
    "VCTKDataModule",
    "CarvanaDataModule",
]