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
