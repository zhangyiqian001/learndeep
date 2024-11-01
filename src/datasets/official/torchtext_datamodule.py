from pathlib import Path

import torch
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import DATASETS
from transformers import AutoTokenizer

from datasets.base_datamodule import BaseDataModule


class TorchTextDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = None
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
            self.num_classes = len(set([label for label, _ in train_iter]))
            self.config.MODEL.ARCH_CONFIG.NUM_CLASSES = self.num_classes

        # BUG 不支持num_workers>0, 预计数据为单个csv，无法多进程
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)

        num_train = int(len(list(train_dataset)) * (1 - self.val_rate))
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
        self.train_set = train_dataset
        self.val_set = split_valid_
        self.test_set = test_dataset
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        inputs = self.tokenizer([text for _, text in batch], padding=True)['input_ids']
        targets = (torch.nn.functional.one_hot(torch.tensor([label - 1 for label, _ in batch]),
                                               num_classes=self.num_classes))
        inputs = torch.tensor(inputs, dtype=torch.int64)
        return inputs.to(self.device), targets.to(self.device)

    def setup(self, stage: str) -> None:
        pass
