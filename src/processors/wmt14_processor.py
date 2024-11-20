import os

import torch
from tokenizers.trainers import BpeTrainer

from datamodules.huggingface_datamodule import HuggingfaceTranslateDataModule
from processors.base_processor import BaseTextProcessor


class WMT14Processor(BaseTextProcessor):
    def __init__(self, *args, file_name=None):
        super().__init__(file_name)

    def train(self, data_iter, file_name, language='fr'):
        trainer = BpeTrainer(vocab_size=32000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.train_from_iterator(map(lambda x: x[language], data_iter), trainer, length=len(data_iter))
        self.tokenizer.save(str(self.cache_dir / file_name))


if __name__ == '__main__':
    # a = pq.ParquetFile("../data/wmt14/fr-en/train-00000-of-00030.parquet")
    # print(a.read().to_pandas().iloc[0])
    # exit()
    kwargs = {
        "path": "../data/wmt14",
        "name": "fr-en",
        "split": None,
        # "cache_dir": "..",
        "batch_size": 16,
        # "processor_src": {"file_name", None},
        # "processor_tgt": {"file_name", None},
        "num_workers": 16
    }

    datamodule = HuggingfaceTranslateDataModule(kwargs)
    datamodule.prepare_data()
    # language = 'fr'
    # WMT14Processor().train(datamodule.train_set['translation'], f"wmt14_{language}.json", language)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

# from conf import *
# from util.data_loader import DataLoader
# from util.tokenizer import Tokenizer
#
# tokenizer = Tokenizer()
# loader = DataLoader(ext=('.en', '.de'),
#                     tokenize_en=tokenizer.tokenize_en,
#                     tokenize_de=tokenizer.tokenize_de,
#                     init_token='<sos>',
#                     eos_token='<eos>')
#
# train, valid, test = loader.make_dataset()
# loader.build_vocab(train_data=train, min_freq=2)
# train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
#                                                      batch_size=batch_size,
#                                                      device=device)
#
# src_pad_idx = loader.source.vocab.stoi['<pad>']
# trg_pad_idx = loader.target.vocab.stoi['<pad>']
# trg_sos_idx = loader.target.vocab.stoi['<sos>']
#
# enc_voc_size = len(loader.source.vocab)
# dec_voc_size = len(loader.target.vocab)
