import glob
import os

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from datamodules.base_datamodule import BaseDataModule


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


class CatsAndDogsDataModule(BaseDataModule):
    def __init__(self, kwargs, ):
        super().__init__(
            kwargs.pop('batch_size'),
            kwargs.pop('num_workers')
        )
        self.kwargs = kwargs

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        train_list = glob.glob(os.path.join(self.kwargs['root'], 'train', '*.jpg'))
        test_list = glob.glob(os.path.join(self.kwargs['root'], 'test', '*.jpg'))
        labels = [path.split('/')[-1].split('.')[0] for path in train_list]
        train_list, valid_list = train_test_split(train_list,
                                                  test_size=0.2,
                                                  stratify=labels,
                                                  random_state=42)
        train_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        self.train_set = CatsDogsDataset(train_list, transform=train_transforms)
        self.val_set = CatsDogsDataset(valid_list, transform=test_transforms)
        self.test_set = CatsDogsDataset(test_list, transform=test_transforms)
