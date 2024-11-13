import math
from typing import Tuple

import torch
from torch import Tensor, nn

from models.base_module import BaseModelModule


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            emb_size,
            dropout,
            maxlen=5000
    ):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


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
    ):
        super().__init__()
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
            dropout=dropout
        )

        # self.ff = nn.Linear(embed_size, num_classes)

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src_emb = self.pos_enc(self.src_embedding(inputs))
        tgt_emb = self.pos_enc(self.tgt_embedding(target))

        outs = self.transformer(
            src_emb,
            tgt_emb,
            # src_mask,
            # tgt_mask,
            # None,
            # src_padding_mask,
            # tgt_padding_mask,
            # memory_key_padding_mask
            src=None,
            tgt=None,
            src_mask=None,
            tgt_mask=None,
            memory_mask=None,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
        )

        return self.ff(outs)


class TransformerModule(BaseModelModule):
    def __init__(self):
        super().__init__()
        self.model = TransformerModel()
