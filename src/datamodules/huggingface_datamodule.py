from datasets import load_dataset

from datamodules.base_datamodule import BaseDataModule


class HuggingfaceDataModule(BaseDataModule):
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
        self.data = load_dataset(**self.kwargs)
        self.train_set, self.val_set, self.test_set = self.data['train'], self.data['validation'], self.data['test']


class HuggingfaceTranslateDataModule(HuggingfaceDataModule):
    def __init__(self, kwargs):
        super().__init__(kwargs)

    # def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
    #     if self.processor_src is not None and self.processor_tgt is not None:
    #         input_ids = self.processor_src(batch['translation']['en'])
    #         target_ids = self.processor_tgt(batch['translation']['fr'])
    #         return {
    #             "inputs": torch.tensor(list(map(lambda x: (x.ids), input_ids))),
    #             "targets": torch.tensor(list(map(lambda x: (x.ids), target_ids)))
    #         }
    #     else:
    #         raise Exception("input and target must pass processor")


if __name__ == '__main__':
    kwargs = {
        "path": "CSTR-Edinburgh/vctk",
        "batch_size": 16,
        "num_workers": 16,
        "cache_dir": "../data"
    }
    datamodule = HuggingfaceDataModule(kwargs)
    datamodule.setup("fit")
    # kwargs = {
    #     "path": "../data/wmt14",
    #     "name": "fr-en",
    #     "split": None,
    #     # "cache_dir": "..",
    #     "batch_size": 16,
    #     # "processor_src": {"file_name", None},
    #     # "processor_tgt": {"file_name", None},
    #     "num_workers": 16
    # }
    # datamodule = HuggingfaceTranslateDataModule(kwargs)
    # datamodule.prepare_data()
    # en_list, fr_list = [], []
    # for item in datamodule.train_set['translation'][:10]:
    #     en_list.append(item['en'])
    #     fr_list.append(item['fr'])
    #     print(item)
    # tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    # from tokenizers.pre_tokenizers import Whitespace
    # tokenizer.pre_tokenizer = Whitespace()
    # tokenizer.train_from_iterator(en_list, trainer)
    # tokenizer.save("tokenizer-wiki.json")
    # tokenizer = tokenizer.from_file("tokenizer-wiki.json")
    # print(tokenizer.get_vocab())
    # print(tokenizer.encode(en_list[0]))
