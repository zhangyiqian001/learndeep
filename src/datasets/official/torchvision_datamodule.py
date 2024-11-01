from pathlib import Path

import torch
from torch.utils.data.dataset import random_split
from torchvision.datasets import *
from torchvision.datasets import MovingMNIST
from transformers import AutoTokenizer

from datasets.base_datamodule import BaseDataModule


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
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.device = config.DEVICE[0]
        self.name = config.DATASET.NAME
        self.data_dir = Path(config.ROOT) / "data" / config.TASK
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.DATASET.TOKENIZER_NAME,
            cache_dir=Path(config.ROOT) / "cache"
        )
        self.config.MODEL.ARCH_CONFIG.INPUT_VOCAB_SIZE = self.tokenizer.vocab_size
        self.val_rate = config.DATASET.VAL_RATE

    def prepare_data(self) -> None:
        train_iter, test_iter = DATASETS[self.name](self.data_dir)
        if "classification" in self.config.TASK:
            num_classes = len(set([label for label, _ in train_iter]))
            self.config.MODEL.ARCH_CONFIG.NUM_CLASSES = num_classes


        num_train = int(len(list(train_dataset)) * (1 - self.val_rate))
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
        self.train_set = train_dataset
        self.val_set = split_valid_
        self.test_set = test_dataset
        self.collate_fn = _collate_fn

    def setup(self, stage: str) -> None:
        pass
