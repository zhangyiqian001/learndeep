from pathlib import Path

import datasets

from datamodules.base_datamodule import BaseDataModule
from datasets import inspect_dataset, load_dataset_builder


class ModelscopeDataModule(BaseDataModule):
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
        inspect_dataset(self.name, self.cache_dir)
        builder = load_dataset_builder(
            str(Path(self.cache_dir) / "wmt_utils.py"),
            language_pair=("fr", "de"),
            subsets={
                datasets.Split.TRAIN: ["commoncrawl_frde"],
                datasets.Split.VALIDATION: ["euelections_dev2019"],
            },
        )

        # Standard version
        builder.download_and_prepare()
        ds = builder.as_dataset()
        print(ds)
        # Streamable version
        # ds = builder.as_streaming_dataset()

    def setup(self, stage: str) -> None:
        pass
        # num_train = int(len(self.train_iter) * (1 - self.val_rate))
        # num_val = int(len(self.train_iter) * self.val_rate)
        # split_train_, split_valid_ = random_split(self.train_iter, [num_train, num_val])
        # self.train_set = split_train_
        # self.val_set = split_valid_
        # self.test_set = self.test_iter


if __name__ == '__main__':
    ModelscopeDataModule("wmt14").prepare_data()
