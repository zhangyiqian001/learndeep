import librosa
import numpy as np
from datasets import load_dataset_builder
from torch.utils.data import Dataset

from datamodules.base_datamodule import BaseDataModule


# 数据集地址：https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
def one_hot_encode(data, channels=256):
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data.ravel()] = 1

    return one_hot


def mu_law_encode(audio, quantization_channels=256):
    """
    Quantize waveform amplitudes.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)
    quantize_space = np.linspace(-1, 1, quantization_channels)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    quantized = np.digitize(quantized, quantize_space) - 1

    return quantized


class VCTKDataset(Dataset):
    def __init__(self, dataset, processor=None):
        self.dataset = dataset
        self.processor = processor

    def __getitem__(self, item):
        if self.processor:
            data = self.processor(self.dataset.data['audio'][item]['path'].as_py())
        else:
            data, _ = librosa.load(self.dataset.data['audio'][item]['path'].as_py(), sr=48000)
            data = mu_law_encode(data)
            data = one_hot_encode(data)
        return data

    def __len__(self):
        return self.dataset.num_rows


class VCTKDataModule(BaseDataModule):
    def __init__(self, kwargs, ):
        super().__init__(
            kwargs.pop('batch_size'),
            kwargs.pop('num_workers')
        )
        self.kwargs = kwargs

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        self.builder = load_dataset_builder(**self.kwargs)
        self.builder.download_and_prepare()
        self.dataset = self.builder.as_dataset(split='train')
        # print(self.dataset.values())
        # print(self.builder.info.features)
        self.train_set = VCTKDataset(self.dataset)
        # self.train_set, self.val_set = random_split(self.train_set, [0.8, 0.2])
        print(self.train_set[0].shape)


if __name__ == '__main__':
    kwargs = {
        "path": "../data/VCTK-Corpus-0.92/vctk.py",
        "batch_size": 16,
        "num_workers": 16,
    }
    datamodule = VCTKDataModule(kwargs)
    datamodule.setup("fit")
