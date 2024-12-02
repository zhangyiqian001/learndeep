from torchtext.datasets import DATASETS

from datamodules.base_datamodule import BaseDataModule


class TorchTextDataModule(BaseDataModule):
    def __init__(
            self,
            name,
            cache_dir='data',
            batch_size=32,
            num_workers=4,
    ):
        super().__init__(
            batch_size,
            num_workers
        )
        self.name = name
        self.cache_dir = cache_dir

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        self.train_set = DATASETS[self.name](self.cache_dir, split='train')
        self.test_set = DATASETS[self.name](self.cache_dir, split='test')
        # self.val_set = DATASETS[self.name](self.cache_dir, split='valid')
