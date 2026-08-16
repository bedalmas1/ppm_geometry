"""Controlled Transformer (Family B): B1 next-event + B2 full-suffix.

Unlike every Family A model in this roster, neither variant here is
adapted from a published repo - both are this project's own from-scratch,
controlled architecture (spec §5: same encoder/embedding-dim/training-budget
across objectives, only the objective varied). Built directly on
`torch.nn.TransformerEncoder`/`torch.nn.TransformerDecoder` (standard,
library-provided Vaswani et al. stacks) rather than a hand-rolled or
paper-specific attention implementation, so the whole model stays genuinely
under this project's control rather than inheriting any one baseline's
idiosyncrasies.

`ControlledTransformerEncoder` is the SHARED component both
`ControlledTransformerNextEvent` (B1) and `ControlledTransformerSuffix`
(B2) reuse UNCHANGED - only the prediction head differs. Its
hyperparameters (configs/models/controlled_transformer_next.yaml,
mirrored verbatim in configs/models/controlled_transformer_suffix.yaml)
are what make the B1-vs-B2 comparison genuinely controlled and
objective-only, per spec §5.

Self-attention in the encoder is bidirectional (no causal mask) over the
observed prefix only - exactly like A1 ProcessTransformer's and A4
SuTraN's own prefix encoders. There is no future leakage: the model is
only ever given the prefix itself, never anything beyond it, so attending
in both directions within that prefix is standard next-event modeling
practice, not a shortcut.

z_t extraction point (Phase 4): `ControlledTransformerEncoder.encode(...)`'s
per-position hidden states - gathered at each sequence's own last
non-padded position for B1's next-event bottleneck
(`gather_last_valid`/`encode_zt`), or consumed directly as the decoder's
cross-attention memory for B2's full-suffix generation
(`ControlledTransformerSuffix.encode`) - the same underlying computation
both objectives read from.
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


class ControlledTransformerSuffix(nn.Module):
    """B2 controlled Transformer (Family B, full-suffix objective).

    Reuses `ControlledTransformerEncoder` UNCHANGED from B1 - only the
    decoder/head differs, keeping the B1-vs-B2 comparison genuinely
    controlled and objective-only (spec §5). The decoder is built the same
    way as the encoder: a standard, library-provided `torch.nn.TransformerDecoder`
    stack, not a hand-rolled or paper-specific implementation (unlike A4
    SuTraN's own vendored decoder in `models/sutran/layers.py`).

    Because `torch.nn.MultiheadAttention`'s own masks (`tgt_mask`,
    `memory_key_padding_mask`) are shaped independently of one another,
    prefix and suffix sequences do NOT need to share one padded window
    length here - unlike A4/A5's adapter, where a hand-rolled attention
    implementation's mask-broadcasting bug forced one shared `window_size`
    (see `models/sutran/adapter.py`'s `get_window_size` docstring). This is
    a genuine simplification enabled by using the standard library, not a
    corner cut. It also means generation can grow the decoder input one
    token at a time (the natural implementation) rather than needing A4's
    fixed-full-window kludge to work around that bug.

    Activity embedding is shared between encoder and decoder (same design
    choice as A4 SuTraN, adopted here on its own merits: halves embedding
    parameters and ties the two vocabularies' geometry together).
    """

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
        self.encoder_module = ControlledTransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, prefix_tokens: torch.Tensor, prefix_pad_mask: torch.Tensor) -> torch.Tensor:
        """z_t for B2: the shared encoder's per-position hidden states -
        the same computation as B1's `ControlledTransformerEncoder.encode`,
        consumed here as the decoder's cross-attention memory instead of
        being pooled to a single vector."""
        return self.encoder_module.encode(prefix_tokens, prefix_pad_mask)

    def decode(
        self, decoder_input_tokens: torch.Tensor, enc_output: torch.Tensor, prefix_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced (training) or single-step (inference) decoding.
        Returns per-position activity-suffix class logits."""
        tgt_len = decoder_input_tokens.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=decoder_input_tokens.device)
        x = self.positional_encoding(self.encoder_module.embedding(decoder_input_tokens))
        hidden = self.decoder(x, enc_output, tgt_mask=causal_mask, memory_key_padding_mask=prefix_pad_mask)
        return self.classifier(hidden)

    def forward(
        self, prefix_tokens: torch.Tensor, prefix_pad_mask: torch.Tensor, decoder_input_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced training forward pass."""
        enc_output = self.encode(prefix_tokens, prefix_pad_mask)
        return self.decode(decoder_input_tokens, enc_output, prefix_pad_mask)

    @torch.no_grad()
    def generate(
        self,
        prefix_tokens: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        sos_token: torch.Tensor,
        max_len: int,
        eos_class: int,
    ) -> torch.Tensor:
        """Greedy autoregressive suffix generation for evaluation (no
        teacher forcing). Grows the decoder input one token at a time -
        the natural implementation, not the fixed-full-window workaround
        A4 SuTraN's `generate()` needs for its own hand-rolled attention
        mask-broadcasting bug (see class docstring).

        Parameters
        ----------
        sos_token : torch.Tensor, shape (batch,)
            The last prefix event's activity token (word-vocab index),
            serving as the decoder's start-of-sequence proxy.
        eos_class : int
            The class index (class-vocab, not word-vocab) representing the
            EOS token - generation for a given instance stops once every
            instance in the batch has predicted it (or `max_len` is
            reached).

        Returns
        -------
        predicted_classes : torch.Tensor, shape (batch, max_len)
            Predicted class indices at each generated step (class-vocab),
            zero-padded past `max_len` if generation stops early via the
            all-finished check. Positions after an instance's own first
            predicted EOS are left as whatever was generated - the caller
            truncates using this, not by re-deriving stopping logic here.
        """
        enc_output = self.encode(prefix_tokens, prefix_pad_mask)
        batch_size = prefix_tokens.size(0)
        device = prefix_tokens.device

        decoder_input = sos_token.view(batch_size, 1)
        predicted_classes = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            logits = self.decode(decoder_input, enc_output, prefix_pad_mask)
            next_class = logits[:, -1, :].argmax(dim=-1)
            predicted_classes[:, step] = next_class
            finished = finished | (next_class == eos_class)
            if finished.all():
                break
            if step + 1 < max_len:
                # class-vocab activity index c (1..V) <-> word-vocab index
                # c+1 (see adapter's Vocab for why this offset is exact);
                # EOS/UNK classes clamp to the word-vocab UNK index (1) so
                # generation stays well-defined for instances not yet
                # finished.
                next_word = torch.where(next_class < eos_class, next_class + 1, torch.ones_like(next_class))
                decoder_input = torch.cat([decoder_input, next_word.view(batch_size, 1)], dim=1)

        return predicted_classes
