import math

import torch
from torch import Tensor, nn
from torchmetrics.text import BLEUScore

from models.base_module import BaseTranslateModelModule


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            emb_size,
            dropout,
            maxlen=40960
    ):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return self.dropout(x + self.pos_embedding[:, :x.size(1), :])


class TransformerModel(nn.Module):
    def __init__(
            self,
            input_vocab_size,
            embed_size,
            dropout=0.1,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            padding_idx=0
    ):
        super().__init__()

        self.padding_idx = padding_idx
        # Output of embedding must be equal (embed_size)
        self.src_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(input_vocab_size, embed_size)

        self.pos_enc = PositionalEncoding(embed_size, dropout)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.ff = nn.Linear(embed_size, input_vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None
        src_emb = self.pos_enc(self.src_embedding(inputs))
        tgt_emb = self.pos_enc(self.tgt_embedding(targets))
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            # src_mask=(inputs != 0),
            # tgt_mask=(targets != 0),
            # memory_mask=None,
            src_key_padding_mask=(inputs == self.padding_idx),
            tgt_key_padding_mask=(targets == self.padding_idx),
            # memory_key_padding_mask=None,
        )
        # return nn.functional.softmax(self.ff(outs), dim=2)
        return self.ff(outs)


class TransformerModule(BaseTranslateModelModule):
    def __init__(
            self,
            input_vocab_size,
            embed_size,
            dropout=0.1,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            padding_idx=0,
            processor_src=None,
            processor_tgt=None,
    ):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=padding_idx)
        # self.loss = nn.CrossEntropyLoss()
        self.metrics = BLEUScore(n_gram=1)
        self.model = TransformerModel(
            input_vocab_size,
            embed_size,
            dropout,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            padding_idx
        )
        self.processor_src = processor_src
        self.processor_tgt = processor_tgt

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if self.processor_src is not None and self.processor_tgt is not None:
            input_ids = self.processor_src(batch['translation']['en'])
            target_ids = self.processor_tgt(batch['translation']['fr'])
            return {
                "inputs": torch.tensor(list(map(lambda x: (x.ids), input_ids))),
                "targets": torch.tensor(list(map(lambda x: (x.ids), target_ids)))
            }
        else:
            raise Exception("input and target must pass processor")
