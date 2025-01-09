from torch.utils.data import Dataset

from datamodules.base_datamodule import BaseDataModule
from datasets import load_dataset_builder

class VCTKDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        pass

    def __len__(self):
        return self.data.num_rows

class VCTKDataModule(BaseDataModule):
    def __init__(self, kwargs, ):
        super().__init__(
            kwargs.pop('batch_size'),
            kwargs.pop('num_workers')
        )
        self.kwargs = kwargs


    def prepare_data(self) -> None:
        pass
        # self.train_set, self.val_set, self.test_set = data['train'], data['validation'], data['test']
        # self.train_set, self.test_set =  data['validation'], data['test']

    def setup(self, stage: str) -> None:
        self.builder = load_dataset_builder(**self.kwargs)
        self.builder.download_and_prepare()
        self.data = self.builder.as_dataset()
        print(self.data.values())
        print(self.builder.info.features)

if __name__ == '__main__':
    kwargs = {
        "path": "../data/VCTK-Corpus-0.92/vctk.py",
        "batch_size": 16,
        "num_workers": 16,
    }
    datamodule = VCTKDataModule(kwargs)
    datamodule.setup("fit")
