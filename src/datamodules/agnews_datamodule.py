from functools import partial

import pandas as pd
from torch.utils.data import Dataset

from datamodules.base_datamodule import BaseDataModule


class AgNewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item][0], self.data[item][0]

    def __len__(self):
        return len(self.data)


class AgNewsDataModule(BaseDataModule):
    def __init__(self, kwargs, ):
        super().__init__(
            kwargs.pop('batch_size'),
            kwargs.pop('num_workers')
        )
        self.root = kwargs['root']

    def prepare_data(self) -> None:
        print("Download and prepare data...")

    def setup(self, stage: str) -> None:
        print("Called at the beginning of fit (train + validate), validate, test, or predict.")
        df = pd.read_csv(f"{self.root}/train.csv", header=None)[[1, 2]]
        test_df = pd.read_csv(f"{self.root}/test.csv", header=None)[[1, 2]]
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
        self.train_set = AgNewsDataset(train_df.values)
        self.val_set = AgNewsDataset(val_df.values)
        self.test_set = AgNewsDataset(test_df.values)
