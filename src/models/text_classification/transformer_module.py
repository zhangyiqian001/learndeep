import math
from typing import Tuple, Any

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.nn import Transformer

from models.base_module import BaseModelModule


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            emb_size,
            dropout,
            maxlen=5000
    ):
        super(PositionalEncoding, self).__init__()
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
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        # Output of embedding must be equal (embed_size)
        self.src_embedding = nn.Embedding(config.INPUT_VOCAB_SIZE, config.EMBED_SIZE)
        # self.tgt_embedding = nn.Embedding(config.INPUT_VOCAB_SIZE, config.EMBED_SIZE)

        self.pos_enc = PositionalEncoding(config.EMBED_SIZE, config.DROPOUT)

        self.transformer = nn.Transformer(
            d_model=config.EMBED_SIZE,
            nhead=config.NUM_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dropout=config.DROPOUT
        )

        self.ff = nn.Linear(config.MAX_LENGTH * config.EMBED_SIZE, config.NUM_CLASSES)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        src_emb = self.pos_enc(self.src_embedding(inputs))
        # tgt_emb = self.pos_enc(self.tgt_embedding(trg))

        outs = self.transformer(
            src_emb,
            src_emb,
            # src_mask,
            # tgt_mask,
            # None,
            # src_padding_mask,
            # tgt_padding_mask,
            # memory_key_padding_mask
        )
        return torch.softmax(self.ff(outs.reshape(inputs.shape[0], -1)), dim=1)


class TransformerModule(BaseModelModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TransformerModel(config.ARCH_CONFIG)
