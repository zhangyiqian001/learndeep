from pathlib import Path
from typing import Any

import torch
from torch.utils.data.dataset import random_split
from torchvision.datasets import *
from torchvision.datasets import MovingMNIST
from torchvision.transforms import transforms

from datasets.base_datamodule import BaseDataModule
from utils.registry import registry

DATASETS = {
    # Image classification
    "Caltech101": Caltech101,
    "Caltech256": Caltech256,
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "Country211": Country211,
    "DTD": DTD,
    "EMNIST": EMNIST,
    "EuroSAT": EuroSAT,
    "FakeData": FakeData,
    "FashionMNIST": FashionMNIST,
    "FER2013": FER2013,
    "FGVCAircraft": FGVCAircraft,
    "Flickr8k": Flickr8k,
    "Flickr30k": Flickr30k,
    "Flowers102": Flowers102,
    "Food101": Food101,
    "GTSRB": GTSRB,
    "INaturalist": INaturalist,
    "ImageNet": ImageNet,
    "KMNIST": KMNIST,
    "LFWPeople": LFWPeople,
    "LSUN": LSUN,
    "MNIST": MNIST,
    "Omniglot": Omniglot,
    "Places365": Places365,
    "PCAM": PCAM,
    "QMNIST": QMNIST,
    "RenderedSST2": RenderedSST2,
    "SEMEION": SEMEION,
    "SBU": SBU,
    "StanfordCars": StanfordCars,
    "STL10": STL10,
    "SUN397": SUN397,
    "SVHN": SVHN,
    "USPS": USPS,

    # Image detection or segmentation
    "CocoDetection": CocoDetection,
    "CelebA": CelebA,
    "Cityscapes": Cityscapes,
    "Kitti": Kitti,
    "OxfordIIITPet": OxfordIIITPet,
    "SBDataset": SBDataset,
    "VOCSegmentation": VOCSegmentation,
    "VOCDetection": VOCDetection,
    "WIDERFace": WIDERFace,

    # Optical Flow
    "FlyingChairs": FlyingChairs,
    "FlyingThings3D": FlyingThings3D,
    "HD1K": HD1K,
    "KittiFlow": KittiFlow,
    "Sintel": Sintel,

    # Stereo Matching
    "CarlaStereo": CarlaStereo,
    "Kitti2012Stereo": Kitti2012Stereo,
    "Kitti2015Stereo": Kitti2015Stereo,
    "CREStereo": CREStereo,
    "FallingThingsStereo": FallingThingsStereo,
    "SceneFlowStereo": SceneFlowStereo,
    "SintelStereo": SintelStereo,
    "InStereo2k": InStereo2k,
    "ETH3DStereo": ETH3DStereo,
    "Middlebury2014Stereo": Middlebury2014Stereo,

    # Image pairs
    "LFWPairs": LFWPairs,
    "PhotoTour": PhotoTour,

    # Image captioning
    "CocoCaptions": CocoCaptions,

    # Video classification
    "HMDB51": HMDB51,
    "Kinetics": Kinetics,
    "UCF101": UCF101,

    # Video prediction
    "MovingMNIST": MovingMNIST,

    # Base classes for custom datasets
    "DatasetFolder": DatasetFolder,
    "ImageFolder": ImageFolder,
    "VisionDataset": VisionDataset,
}

class TorchVisionDataModule(BaseDataModule):
    def __init__(self, config, transfer):
        super().__init__(config, transfer)
        self.num_classes = None
        self.config = config
        self.name = config.NAME.upper()
        self.data_dir = Path(registry.get_path("root")) / "data"
        self.val_rate = config.VAL_RATE

    def prepare_data(self) -> None:
        self.train_iter = DATASETS[self.name](
            self.data_dir, train=True, download=True, transform = transforms.ToTensor(),)
        self.test_iter = DATASETS[self.name](
            self.data_dir, train=False, download=True, transform = transforms.ToTensor(),)
        try:
            self.num_classes = len(self.train_iter.classes)
        except Exception:
            raise Exception("不支持分类模型")

    def setup(self, stage: str) -> None:
        num_train = int(len(self.train_iter) * (1 - self.val_rate))
        num_val = int(len(self.train_iter) * self.val_rate)
        split_train_, split_valid_ = random_split(self.train_iter, [num_train, num_val])
        self.train_set = split_train_
        self.val_set = split_valid_
        self.test_set = self.test_iter

    @classmethod
    def from_config(cls, config, transfer=None):
        return cls(config, transfer)


@registry.register_datamodule("mnist")
class MnistDataModule(TorchVisionDataModule):
    def __init__(self, config, transfer):
        super().__init__(config, transfer)
    @classmethod
    def from_config(cls, config, transfer=None):
        return cls(config, transfer)

@registry.register_datamodule("cifar10")
class CIFAR10DataModule(TorchVisionDataModule):
    def __init__(self, config, transfer):
        super().__init__(config, transfer)
    @classmethod
    def from_config(cls, config, transfer=None):
        return cls(config, transfer)