from .torchvision_datamodule import TorchVisionDataModule, CelebADataModule
from .huggingface_datamodule import HuggingfaceDataModule
from .agnews_datamodule import AgNewsDataModule
from .vctk_datamodule import VCTKDataModule, VCTKMuLawDataModule
from .carvana_datamodule import CarvanaDataModule
from .catsanddogs_datamodule import CatsAndDogsDataModule

__all__ = [
    "TorchVisionDataModule",
    "CelebADataModule",
    "HuggingfaceDataModule",
    "AgNewsDataModule",
    "VCTKDataModule",
    "VCTKMuLawDataModule",
    "CarvanaDataModule",
    "CatsAndDogsDataModule",
]