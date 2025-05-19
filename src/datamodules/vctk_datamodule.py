import glob
import math

import librosa
import string

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset_builder
from librosa import feature
from torch.utils.data import Dataset, random_split

from datamodules.base_datamodule import BaseDataModule
from utils.text import text_to_sequence


# 原数据（16位）onehot=2**16=65536，mu_law将数据量化到classes（256）
# 具体参考wavenet论文
def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    # [i if i != -1 else 0 for i in quantized]
    return np.array([i if i != -1 else 0 for i in quantized])


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def processor_data(sr=16000, classes=256, item_length=80000):
    builder = load_dataset_builder(path="../data/VCTK-Corpus-0.92/vctk.py")
    builder.download_and_prepare()
    dataset = builder.as_dataset(split='train')
    count = 0
    for i, item in enumerate(dataset):
        data, _ = librosa.load(item['file'], sr=sr)
        mu_law = quantize_data(data, classes)
        mu_law = torch.from_numpy(mu_law)
        audio_len = math.floor(len(mu_law) / item_length)
        for j in range(0, audio_len):
            one_hot = torch.FloatTensor(classes, item_length).zero_()
            index = mu_law[j * item_length:(j + 1) * item_length].unsqueeze(0)
            one_hot.scatter_(0, index, 1.)
            np.save(f"../cache/vctk/feature_%05d" % (count + j), one_hot.numpy())
        count += audio_len


# 数据集地址：https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
class VCTKDataset(Dataset):
    def __init__(self, dataset, sr=16000, n_mfcc=80):
        self.dataset = dataset
        self.sr = sr
        self.n_mfcc = n_mfcc

    def __getitem__(self, item):
        data, _ = librosa.load(self.dataset[item]['file'], sr=self.sr)
        # 原数据（16位）onehot=2**16=65536，mu_law将数据量化到classes（256）
        # 具体参考wavenet论文
        mu_law = quantize_data(data, self.classes)
        # (65, n_mfcc)
        mel_feature = librosa.feature.mfcc(
            y=data,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
        ).T
        text = self.dataset[item]['text'].translate(str.maketrans("", "", string.punctuation))
        character = text_to_sequence(text, ['english_cleaners'])
        character = torch.from_numpy(np.array(character))
        mel_target = torch.from_numpy(np.array(mel_feature))
        mu_law = torch.from_numpy(mu_law)
        return {'text': character, 'mel_target': mel_target, "raw_audio": mu_law}

    def __len__(self):
        return len(self.dataset)


class VCTKMuLawDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        mu_law = torch.from_numpy(self.dataset[item*1000:(item+1)*1000])
        return {"mu_law_audio": mu_law}

    def __len__(self):
        return math.ceil(len(self.dataset) / 1000)


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
        builder = load_dataset_builder(**self.kwargs)
        builder.download_and_prepare()
        dataset = builder.as_dataset(split='train')
        train_data, val_data = random_split(dataset, [0.8, 0.2])
        self.train_set = VCTKDataset(train_data)
        self.val_set = VCTKDataset(val_data)
        self.collate_fn = collate_fn
        # print(self.train_set[0])


# 填充列
def collate_fn(batch):
    text_lens = [b['text'].size(0) for b in batch]
    max_text_len = max(text_lens)
    text = torch.stack([F.pad(b['text'], (0, max_text_len - b['text'].shape[0])) for b in batch])

    text_pos = []
    for length_src_row in text_lens:
        text_pos.append(np.pad([i + 1 for i in range(int(length_src_row))],
                               (0, max_text_len - int(length_src_row)), 'constant'))
    text_pos = torch.from_numpy(np.array(text_pos))

    mel_target_lens = [b['mel_target'].size(0) for b in batch]
    max_mel_len = max(mel_target_lens)
    mel_target = torch.stack([F.pad(b['mel_target'], (0, 0, 0, max_mel_len - b['mel_target'].shape[0])) for b in batch])

    mel_pos = []
    for length_mel_row in mel_target_lens:
        mel_pos.append(np.pad([i + 1 for i in range(int(length_mel_row))],
                              (0, max_mel_len - int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    raw_audio_lens = [b['raw_audio'].size(0) for b in batch]
    raw_audio_len = max(raw_audio_lens)
    raw_audio = torch.stack([F.pad(b['raw_audio'], (0, raw_audio_len - b['raw_audio'].shape[0])) for b in batch])

    return {
        "text": text,
        "text_pos": text_pos,
        "mel_target": mel_target,
        "mel_pos": mel_pos,
        "mel_max_len": max_mel_len,
        "raw_audio": raw_audio
    }


class VCTKMuLawDataModule(BaseDataModule):
    def __init__(self, kwargs, ):
        super().__init__(
            kwargs.pop('batch_size'),
            kwargs.pop('num_workers')
        )
        self.kwargs = kwargs

    def prepare_data(self) -> None:
        # processor_data()
        pass

    def setup(self, stage: str) -> None:
        data = np.concatenate([np.load(file_path) for file_path in glob.glob(self.kwargs['cache'])], axis=1).T
        length = len(data)
        train_data = data[:int(length*0.8)]
        val_data = data[int(length*0.8):]
        self.train_set = VCTKMuLawDataset(train_data)
        self.val_set = VCTKMuLawDataset(val_data)


if __name__ == '__main__':
    # kwargs = {
    #     "path": "../data/VCTK-Corpus-0.92/vctk.py",
    #     "batch_size": 64,
    #     "num_workers": 16,
    #     "trust_remote_code": False
    # }
    # datamodule = VCTKDataModule(kwargs)
    # datamodule.setup("fit")
    # print(next(iter(datamodule.train_dataloader())))
    # processor_data()
    kwargs = {
        "cache": "../cache/vctk/*.npy",
        "batch_size": 64,
        "num_workers": 16,
    }
    datamodule = VCTKMuLawDataModule(kwargs)
    datamodule.setup(stage="fit")
