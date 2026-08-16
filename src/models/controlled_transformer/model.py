"""B1 controlled Transformer (Family B, next-event objective).

Unlike every Family A model in this roster, this is not adapted from a
published repo - it is this project's own from-scratch, controlled
architecture (spec §5: same encoder/embedding-dim/training-budget across
objectives, only the objective varied). Built directly on
`torch.nn.TransformerEncoder` (a standard, library-provided Vaswani et al.
encoder stack) rather than a hand-rolled or paper-specific attention
implementation, so the whole model stays genuinely under this project's
control rather than inheriting any one baseline's idiosyncrasies.

`ControlledTransformerEncoder` is the SHARED component B2 (full-suffix,
Family B, to be built next) will reuse unchanged - only the prediction head
differs between B1 (next-event, this file) and B2. Its hyperparameters
(configs/models/controlled_transformer_next.yaml) are the ones that must be
copied verbatim into B2's config for a genuinely controlled,
objective-only comparison.

Self-attention here is bidirectional (no causal mask) over the observed
prefix only - exactly like A1 ProcessTransformer's and A4 SuTraN's own
prefix encoders. There is no future leakage: the model is only ever given
the prefix itself, never anything beyond it, so attending in both
directions within that prefix is standard next-event modeling practice, not
a shortcut.

z_t extraction point (Phase 4): `encode_zt(...)`, the encoder's per-position
hidden states gathered at each sequence's own last non-padded position - the
natural next-event bottleneck, one vector per prefix length in a single
forward pass, analogous to A5/A6/A7's single pooled-per-prefix vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.controlled_transformer.layers import PositionalEncoding

PAD_IDX = 0


class ControlledTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def encode(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, seq_len) token ids. pad_mask: (batch, seq_len),
        True at padded positions (nn.TransformerEncoder's own
        src_key_padding_mask convention). Returns (batch, seq_len, d_model)
        per-position hidden states."""
        x = self.positional_encoding(self.embedding(tokens))
        return self.encoder(x, src_key_padding_mask=pad_mask)


def gather_last_valid(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """hidden: (batch, seq_len, d_model). lengths: (batch,), the number of
    real (non-padded) positions per sequence. Returns (batch, d_model): the
    hidden state at each sequence's own last real position."""
    idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, idx).squeeze(1)


class ControlledTransformerNextEvent(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ControlledTransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def encode_zt(self, tokens: torch.Tensor, pad_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder.encode(tokens, pad_mask)
        return gather_last_valid(hidden, lengths)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        z_t = self.encode_zt(tokens, pad_mask, lengths)
        return self.classifier(z_t)
