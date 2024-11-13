import torch
from lightning.pytorch.cli import LightningCLI
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter.callable import MapperIterDataPipe
from torch.utils.data.datapipes.iter.sharding import ShardingFilterIterDataPipe
from torchmetrics import Accuracy
from torchtext.data import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from transformers import BertTokenizer
from omegaconf import OmegaConf

from models import BaseModelModule, AlexNetModule
from datasets import MnistDataModule
from tasks import create_task
from utils.registry import registry


def a(f):
    print(f)
# def a(batch):
#     print(batch)
#     a, b = batch[0]
#     return a, b
if __name__ == '__main__':
    cli = LightningCLI(AlexNetModule, MnistDataModule)
    # a = torch.tensor(
    #     [
    #         [2.0, 0.5],
    #         [0.5, 2.0]
    #     ]
    # )
    # b = torch.tensor([1, 2])
    # loss = CrossEntropyLoss()
    # print(loss(a, b))
    # config = OmegaConf.load("configs/text_classification/transformer.yaml")
    # print(config.get("BASE", 116))
    # v = {"F": 1}
    # lowercase_students = {key.lower(): value for key, value in v.items()}
    # a(**lowercase_students)
    # train_set, test_set = AG_NEWS()
    # print(train_set)
    # # train_set = to_map_style_dataset(train_set)
    # loader = DataLoader(train_set, num_workers=16, collate_fn=a)
    # for label, text in loader:
    #     print(label, text)
    #     break
    # print(train_set)
    # tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased",)
    # print(torch.tensor(tokenizer([i[1] for i in train_set[:10]], padding=True)['input_ids']).shape)
    # a = tokenizer(train_set[0][1])
    # print(torch.nn.functional.one_hot(torch.tensor([[1], [2]],), num_classes=4))
    # print(a)
    # a = torch.rand(10, 1024)
    # linear = nn.Linear(1024, 4)
    # out = linear(a)
    # # out = torch.softmax(linear(a), dim=1)
    # print(out.shape)
    # loss = nn.CrossEntropyLoss()
    # acc = Accuracy(task="multiclass", num_classes=2)
    # g = loss(torch.tensor([0.2, 0.8]), torch.tensor([1.0, 0.0]))
    # g = acc(torch.tensor([1.0, 0.1]).argmax(1), torch.tensor([1.0]))
    # print(g)
    # print(torch.tensor([[0.9, 0.1],
    #                     [0.8, 0.2]]).argmax(1))


