from typing import Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):

    def __init__(self, config, transfer):
        super().__init__()

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None
        self.transfer = transfer

        self.batch_size = config.BATCH_SIZE
        self.num_workers = config.NUM_WORKERS

        self.persistent_workers = True
        # 如果数据集大小不能被批处理大小整除,删除最后一个未完成的批
        self.drop_last = True


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.transfer is None:
            return {"inputs": batch[0], "targets": batch[1]}
        else:
            return {"inputs": self.transfer(batch[0]), "targets": batch[1]}

    @classmethod
    def from_config(cls, config):
        return cls(config)
