import torch
from torch.utils.data.dataset import random_split, Dataset
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import DATASETS

from datamodules.base_datamodule import BaseDataModule


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
        targets = (torch.nn.functional.one_hot(torch.tensor(self.data[index][0] - 1), num_classes=self.num_classes))
        inputs = {key: torch.tensor(value, dtype=torch.int64).to(self.device) for key, value in inputs.items()}
        targets = torch.tensor(targets, dtype=torch.float32)
        return inputs, targets.to(self.device)

    def __len__(self):
        return len(self.data)


class TorchTextDataModule(BaseDataModule):
    def __init__(
            self,
            name,
            val_rate=0.2,
            cache_dir='data',
            batch_size=32,
            num_workers=4,
    ):
        super().__init__(
            batch_size,
            num_workers
        )
        self.name = name
        self.val_rate = val_rate
        self.cache_dir = cache_dir

    def prepare_data(self) -> None:
        self.train_iter = DATASETS[self.name](self.cache_dir, split='train')
        self.test_iter = DATASETS[self.name](self.cache_dir, split='test')

        # BUG 不支持num_workers>0, 预计数据为单个csv，无法多进程
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)

        num_train = int(len(list(train_dataset)) * (1 - self.val_rate))
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
        self.train_set = train_dataset
        self.val_set = split_valid_
        self.test_set = test_dataset

    def setup(self, stage: str) -> None:
        pass
