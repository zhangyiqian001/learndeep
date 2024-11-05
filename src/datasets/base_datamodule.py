from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class BaseDataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None

        self.batch_size = config.DATASET.BATCH_SIZE
        self.num_workers = config.DATASET.NUM_WORKERS

        self.persistent_workers = True
        # 如果数据集大小不能被批处理大小整除,删除最后一个未完成的批
        self.drop_last = True
        self.collate_fn = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn
        )
