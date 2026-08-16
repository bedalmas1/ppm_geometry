"""Standard sinusoidal positional encoding (Vaswani et al. 2017), batch_first
- shared by every Family B controlled-Transformer variant (B1 here, B2
later). This is the textbook formula, not vendored from any specific
baseline paper's repo - the encoder stack itself is built directly on
torch.nn.TransformerEncoder (see model.py), keeping this whole model
genuinely this project's own controlled architecture.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be an even number"
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)
