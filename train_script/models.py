import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension
        self.encoding = self.encoding.permute(
            1, 0, 2
        )  # (max_len, batch_size, d_model)
        self.register_buffer("pe", self.encoding)

    def forward(self, x):
        """
        x: Tensor of shape (seq_len, batch_size, d_model)
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_max_len,
        tgt_max_len,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self.src_pos_enc = PositionalEncoding(d_model, src_max_len)
        self.tgt_pos_enc = PositionalEncoding(d_model, tgt_max_len)

    def forward(self, src, tgt):
        src = self.src_embedding(src) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32)
        )
        tgt = self.tgt_embedding(tgt) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32)
        )
        src = src.permute(1, 0, 2)  # (S, N, E)
        tgt = tgt.permute(1, 0, 2)  # (T, N, E)
        src = src + self.src_pos_enc(src)
        tgt = tgt + self.tgt_pos_enc(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0))
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output
