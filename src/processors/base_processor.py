import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class BaseProcessor:
    def __init__(self):
        self.cache_dir = Path("cache")

    def __call__(self, *args, **kwargs):
        pass


class BaseTextProcessor(BaseProcessor):
    def __init__(self, file_name=None):
        super().__init__()
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        if file_name is not None:
            self.file_name = str(self.cache_dir / file_name)
            self.tokenizer = self.tokenizer.from_file(self.file_name)

    def __call__(self, row_iter):
        # ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing
        return self.tokenizer.encode_batch(row_iter)

    def train(self, data, file_name):
        trainer = BpeTrainer(vocab_size=32000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.train_from_iterator(map(lambda x: x['fr'], data), trainer, length=len(data))
        # self.tokenizer.train(batch_iterator(data), trainer)
        self.tokenizer.save(str(self.cache_dir / file_name))

