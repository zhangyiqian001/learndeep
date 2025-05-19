import torch
from torch import nn, Tensor

from modules.base_module import BaseTranslateModule
from modules.common_module import PositionalEncoding


class BertModel(nn.Module):
    def __init__(
            self,
            input_vocab_size,
            embed_size,
            dropout=0.1,
            num_heads=8,
            num_decoder_layers=12,
            dim_feedforward=2048,
            padding_idx=0
    ):
        super().__init__()

        self.padding_idx = padding_idx
        # Output of embedding must be equal (embed_size)
        self.src_embedding = nn.Embedding(input_vocab_size, embed_size)

        self.pos_enc = PositionalEncoding(embed_size, dropout)

        self.bert = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=embed_size,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_decoder_layers)
            ]
        )

        self.ff = nn.Linear(embed_size, input_vocab_size)

    def forward(self, inputs: Tensor) -> Tensor:
        src_emb = self.pos_enc(self.src_embedding(inputs))

        src_pad_mask = (inputs == self.padding_idx)

        outs = self.bert(
            src=src_emb,
            src_key_padding_mask=src_pad_mask,
        )
        return self.ff(outs)


class BertModule(BaseTranslateModule):
    def __init__(
            self,
            input_vocab_size,
            embed_size,
            dropout=0.1,
            num_heads=8,
            num_decoder_layers=6,
            dim_feedforward=2048,
            padding_idx=0,
            processor_src=None,
            processor_tgt=None,
            loss=None,
            metrics=None
    ):
        super().__init__()
        self.loss = loss
        self.metrics = metrics
        self.model = BertModel(
            input_vocab_size,
            embed_size,
            dropout,
            num_heads,
            num_decoder_layers,
            dim_feedforward,
            padding_idx
        )
        self.processor_src = processor_src
        self.processor_tgt = processor_tgt

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if self.processor_src is not None and self.processor_tgt is not None:
            # input_ids = self.processor_src(batch['translation']['en'])
            # target_ids = self.processor_tgt(batch['translation']['fr'])
            input_ids = self.processor_src(batch[0])
            target_ids = self.processor_tgt(batch[1])
            return {
                "inputs": torch.tensor(list(map(lambda x: (x.ids), input_ids))),
                "targets": torch.tensor(list(map(lambda x: (x.ids), target_ids)))
            }
        else:
            raise Exception("input and target must pass processor")
