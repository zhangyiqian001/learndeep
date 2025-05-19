import math

import torch
from torch import Tensor, nn

from modules.base_module import BaseTranslateModule
from modules.common_module import PositionalEncoding


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

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        src_emb = self.pos_enc(self.src_embedding(inputs))
        tgt_emb = self.pos_enc(self.tgt_embedding(targets))

        src_pad_mask = (inputs == self.padding_idx)
        tgt_pad_mask = (targets == self.padding_idx)
        tgt_sub_mask = nn.Transformer.generate_square_subsequent_mask(targets.shape[1], device=targets.device)

        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            # src_mask=(inputs != 0),
            tgt_mask=tgt_sub_mask,
            # memory_mask=None,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            # memory_key_padding_mask=None,
        )
        return self.ff(outs)


class TransformerModule(BaseTranslateModule):
    def __init__(
            self,
            model,
            loss,
            metrics,
            processor_src,
            processor_tgt,
    ):
        super().__init__()
        self.loss = loss
        self.metrics = metrics
        self.model = model
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
