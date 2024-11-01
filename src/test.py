import torch
from torch import nn
from torch.utils.data.datapipes.iter.callable import MapperIterDataPipe
from torch.utils.data.datapipes.iter.sharding import ShardingFilterIterDataPipe
from torchtext.data import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from transformers import BertTokenizer
if __name__ == '__main__':
    ShardingFilterIterDataPipe
    train_set, test_set = AG_NEWS()
    print(train_set.source_datapipe)
    train_set = to_map_style_dataset(train_set)
    MapperIterDataPipe
    print(train_set)
    # tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased",)
    # print(torch.tensor(tokenizer([i[1] for i in train_set[:10]], padding=True)['input_ids']).shape)
    # a = tokenizer(train_set[0][1])
    # print(torch.nn.functional.one_hot(torch.tensor(train_set[0][0]), num_classes=4))
    # print(a)
    # a = torch.rand(10, 1024)
    # linear = nn.Linear(1024, 4)
    # out = linear(a)
    # print(out.shape)
    # loss = nn.CrossEntropyLoss()
    # g = loss(out, torch.rand(10, 4).view(-1))
    # print(g)


