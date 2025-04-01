from lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):

    def __init__(
            self,
            batch_size,
            num_workers
    ):
        super().__init__()

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transfer = None
        # 缓存workers
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

    # def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
    #     if dataloader_idx == 0:
    #         print(batch)
    #     if self.transfer is None:
    #         return {"inputs": batch[0], "targets": batch[1]}
    #     else:
    #         return {"inputs": self.transfer(batch[0]), "targets": batch[1]}
