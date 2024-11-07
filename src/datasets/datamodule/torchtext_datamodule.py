from pathlib import Path

import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset
from torchtext.datasets import DATASETS
from transformers import AutoTokenizer
from datasets.base_datamodule import BaseDataModule


class TextClassificationDataset(Dataset):
    def __init__(self, iter_data, tokenizer, max_length, num_classes, device):
        self.device = device
        self.num_classes = num_classes
        self.data = list(iter_data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        inputs = self.tokenizer(
            self.data[index][1],
            padding="max_length",
            max_length=self.max_length
        )
        targets = (torch.nn.functional.one_hot(torch.tensor(self.data[index][0]-1), num_classes=self.num_classes))
        inputs = {key: torch.tensor(value, dtype=torch.int64).to(self.device) for key, value in inputs.items()}
        targets = torch.tensor(targets, dtype=torch.float32)
        return inputs, targets.to(self.device)

    def __len__(self):
        return len(self.data)


class TorchTextDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        config.MODEL.ARCH_CONFIG.MAX_LENGTH = config.DATASET.MAX_LENGTH
        self.num_classes = None
        self.config = config
        self.device = config.BASE.DEVICE[0]
        self.name = config.DATASET.NAME
        self.max_length = config.DATASET.MAX_LENGTH
        self.data_dir = Path(config.BASE.ROOT) / "data" / config.BASE.TASK
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.DATASET.TOKENIZER_NAME,
            cache_dir=Path(config.BASE.ROOT) / "cache"
        )
        self.config.MODEL.ARCH_CONFIG.INPUT_VOCAB_SIZE = self.tokenizer.vocab_size
        self.val_rate = config.DATASET.VAL_RATE

    def prepare_data(self) -> None:
        self.train_iter, self.test_iter = DATASETS[self.name](self.data_dir)
        if "classification" in self.config.BASE.TASK:
            self.num_classes = len(set([label for label, _ in self.train_iter]))
            self.config.MODEL.NUM_CLASSES = self.num_classes

    def setup(self, stage: str) -> None:
        train_dataset = TextClassificationDataset(self.train_iter, self.tokenizer, self.max_length, self.num_classes,
                                                  self.device)
        test_dataset = TextClassificationDataset(self.test_iter, self.tokenizer, self.max_length, self.num_classes,
                                                 self.device)
        num_train = int(len(train_dataset) * (1 - self.val_rate))
        num_val = int(len(train_dataset) * self.val_rate)
        split_train_, split_valid_ = random_split(train_dataset, [num_train, num_val])
        self.train_set = split_train_
        self.val_set = split_valid_
        self.test_set = test_dataset

