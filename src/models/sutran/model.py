"""SuTraN (A4) model architecture - activity-only full-suffix scope.

Adapted from Wuyts & De Weerdt, "SuTraN: an Encoder-Decoder Transformer for
Full-Context-Aware Suffix Prediction of Business Processes" (ICPM 2024),
https://github.com/BrechtWts/SuffixTransformerNetwork (MIT). See
paper/related_work_model_audit.md Section B and
paper/phase3_baseline_reproduction.md for the full audit/reproduction notes.

**Scope decision, applied consistently across the whole model roster**: the
original SuTraN (both its DA and NDA variants) is a multi-task model,
jointly predicting the activity suffix, the timestamp ("time till next
event") suffix, and a scalar remaining-runtime value. This project
implements **activity-suffix prediction only** - not just dropping the
output heads, but removing timestamp features from the decoder's input
entirely. This is a stricter simplification than "NDA" (non-data-aware,
which still uses 2 numeric time proxies as decoder input alongside
activity), and was chosen deliberately rather than defaulting to NDA:
NDA's timestamp *inputs* only make sense at inference time if the model
also *predicts* the next timestamp (autoregressively feeding its own
timestamp prediction back in as the next step's decoder input) - since
this project drops the timestamp-prediction head (consistent with A1
ProcessTransformer and A2 GenerativeLSTM both being scoped to activity-only
prediction, dropping their own time/role heads), keeping timestamp *inputs*
without a way to generate them at inference would be architecturally
inconsistent. Every model in this roster therefore predicts (and consumes)
activities only - a deliberately homogeneous prediction-target family across
the whole roster, not an per-model ad-hoc choice.

Hyperparameters (d_model=32, num_prefix_encoder_layers=4,
num_decoder_layers=4, num_heads=8, d_ff=128, dropout=0.2) are taken directly
from the repo's own TRAIN_EVAL_SUTRAN_NDA.py, not guessed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.sutran.layers import DecoderLayer, EncoderLayer, PositionalEncoding

PAD_IDX = 0


class SuTraNActivityOnly(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 32,
        num_prefix_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        # Activity embedding shared between encoder and decoder (matches the
        # original repo's own design choice).
        self.act_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_prefix_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )
        self.fc_out_act = nn.Linear(d_model, num_classes)

    def encode(self, prefix_tokens: torch.Tensor, prefix_pad_mask: torch.Tensor) -> torch.Tensor:
        """Returns the per-prefix-position encoder representation
        (batch, prefix_len, d_model) - this project's z_t extraction point
        for SuTraN (Phase 4), identified during the Phase 1 audit as the
        cleanest across every model: one vector per prefix length, in a
        single forward pass, before any prediction head."""
        x = self.positional_encoding(self.act_emb(prefix_tokens))
        for layer in self.encoder_layers:
            x = layer(x, prefix_pad_mask)
        return x

    def decode(self, decoder_input_tokens: torch.Tensor, enc_output: torch.Tensor, prefix_pad_mask: torch.Tensor) -> torch.Tensor:
        """Teacher-forced (training) or single-step (inference) decoding.
        Returns per-position activity-suffix class logits."""
        x = self.positional_encoding(self.act_emb(decoder_input_tokens))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, prefix_pad_mask)
        return self.fc_out_act(x)

    def forward(self, prefix_tokens, prefix_pad_mask, decoder_input_tokens):
        """Teacher-forced training forward pass."""
        enc_output = self.encode(prefix_tokens, prefix_pad_mask)
        return self.decode(decoder_input_tokens, enc_output, prefix_pad_mask)

    @torch.no_grad()
    def generate(self, prefix_tokens, prefix_pad_mask, sos_token: torch.Tensor, max_len: int, eos_class: int):
        """Greedy autoregressive suffix generation for evaluation (no
        teacher forcing - the ground-truth suffix is not available at
        inference time, exactly as in real deployment).

        Uses a FIXED-SHAPE decoder input of length `max_len` at every step
        (unfilled future positions left as PAD/0) rather than growing the
        sequence one token at a time. This isn't just an efficiency choice:
        the cross-attention mask (derived from the encoder/prefix padding
        mask) is only broadcastable when the decoder's query length matches
        the length it was built for. A naively-growing decoder input would
        have query length 1, 2, 3, ... while the mask always has length
        `max_len`, breaking the broadcast (confirmed by a smoke test that
        crashed with exactly this shape mismatch). Padding the query to a
        fixed `max_len` throughout - reading out only the newly-decodable
        position via the causal self-attention mask each step - sidesteps
        the issue entirely, since a position's causal mask only depends on
        the (already-filled) positions at or before it.

        Parameters
        ----------
        sos_token : torch.Tensor, shape (batch,)
            The last prefix event's activity token (word-vocab index),
            serving as the decoder's start-of-sequence proxy per the
            original repo's own convention.
        eos_class : int
            The class index (class-vocab, not word-vocab) representing the
            EOS token - generation for a given instance stops once this
            class is predicted.

        Returns
        -------
        predicted_classes : torch.Tensor, shape (batch, max_len)
            Predicted class indices at each generated step (class-vocab).
            Positions after an instance's first predicted EOS are left as
            whatever was generated (the caller truncates using this, not by
            re-deriving stopping logic here).
        """
        enc_output = self.encode(prefix_tokens, prefix_pad_mask)
        batch_size = prefix_tokens.size(0)
        device = prefix_tokens.device

        decoder_input = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        decoder_input[:, 0] = sos_token
        predicted_classes = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

        for step in range(max_len):
            logits = self.decode(decoder_input, enc_output, prefix_pad_mask)  # (batch, max_len, num_classes)
            next_class = logits[:, step, :].argmax(dim=-1)  # (batch,) - prediction FOR position `step`
            predicted_classes[:, step] = next_class
            if step + 1 < max_len:
                # class-vocab activity index c (1..V) <-> word-vocab index
                # c+1 (see adapter.py's Vocab for why this offset is exact);
                # EOS/UNK classes clamp to the word-vocab UNK index (1) so
                # generation stays well-defined for instances not yet
                # finished (their true stopping point is truncated later by
                # the caller, based on the first predicted EOS).
                next_word = torch.where(next_class < eos_class, next_class + 1, torch.ones_like(next_class))
                decoder_input[:, step + 1] = next_word

        return predicted_classes
