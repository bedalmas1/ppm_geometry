"""CRTP-LSTM (A5) model architecture - activity-only, direct (non-
autoregressive) full-suffix prediction.

Adapted from Gunnarsson, vanden Broucke & De Weerdt, "A Direct Data Aware
LSTM Neural Network Architecture for Complete Remaining Trace and Runtime
Prediction," IEEE Transactions on Services Computing 16(4), 2023, as
re-implemented in https://github.com/BrechtWts/SuffixTransformerNetwork
(MIT, same repo as A4 SuTraN). See paper/related_work_model_audit.md
Section B and paper/phase3_baseline_reproduction.md for the full audit.

**Mechanism (why no autoregressive decoding is needed, unlike A4 SuTraN)**:
the prefix is fed to a BIDIRECTIONAL LSTM as a **left-padded** sequence -
real prefix events pinned to the END of the fixed `window_size` window,
padding at the start (see adapter.py's `left_pad`). Because the LSTM is
bidirectional, every output position has access to the entire prefix via
the backward pass, regardless of how short the real prefix is. This lets
the model directly regress onto ALL suffix positions in a single forward
pass, with output position i trained against the i-th future event
(right-padded suffix target, i=0 is the first future event) - no
step-by-step generation loop, unlike SuTraN's autoregressive decoder.

**Scope decision (consistent with A1/A2/A4)**: activity-only. The original
architecture has a second dedicated LSTM branch + head predicting a
remaining-runtime suffix; this project drops it, matching A4's own
activity-only scoping and keeping every roster model's prediction target
homogeneous. Also drops the 2 numeric time-proxy features the repo's own
"NDA" (no-context) variant still keeps as decoder... er, encoder input
(stricter than NDA, same rationale as A4: since we don't predict
timestamps, we don't need timestamp-derived case/event features as input
either - the roster is activity-in, activity-out throughout).

Hyperparameters (d_model=80, dropout=0.2, num_shared_LSTMlayers=1,
num_dedicated_LSTMlayers=1) are taken directly from the repo's own
TRAIN_EVAL_CRTP_LSTM_ND.py, not guessed. The embedding-size formula
(`min(600, round(1.6 * n**0.56))`) is the repo's own formula, applied here
to this project's own (smaller) activity vocabulary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


def embedding_size_for_cardinality(n: int) -> int:
    return min(600, round(1.6 * n**0.56))


class CRTPLSTMActivityOnly(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 80,
        dropout: float = 0.2,
        num_shared_lstm_layers: int = 1,
        num_dedicated_lstm_layers: int = 1,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be an even number"
        self.hidden_size = d_model // 2
        self.d_model = d_model

        embed_dim = embedding_size_for_cardinality(vocab_size)
        self.act_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm_shared = nn.LSTM(
            input_size=embed_dim, hidden_size=self.hidden_size, num_layers=num_shared_lstm_layers,
            batch_first=True, bidirectional=True,
        )
        self.bn_shared = nn.BatchNorm1d(d_model)

        self.lstm_act = nn.LSTM(
            input_size=d_model, hidden_size=self.hidden_size, num_layers=num_dedicated_lstm_layers,
            batch_first=True, bidirectional=True,
        )
        self.bn_act = nn.BatchNorm1d(d_model)
        self.fc_out_act = nn.Linear(d_model, num_classes)

        self._reset_lstm_parameters()

    def _reset_lstm_parameters(self) -> None:
        """Glorot/orthogonal init, matching the original repo's own
        `reset_parameters_lstm` (mimicking Keras' glorot_uniform default,
        used by the original CRTP-LSTM's TensorFlow implementation)."""
        for lstm in (self.lstm_shared, self.lstm_act):
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    init.orthogonal_(param.data)

    def encode(self, prefix_tokens_left_padded: torch.Tensor) -> torch.Tensor:
        """Returns the shared BiLSTM's output (batch, window_size, d_model).
        This project's z_t extraction point for CRTP-LSTM (Phase 4): because
        the input is LEFT-padded, position `window_size - 1` always
        corresponds to where the last real prefix event sits - the natural
        one-vector-per-prefix summary representation, extracted the same
        way regardless of the actual prefix length."""
        x = self.dropout(self.act_emb(prefix_tokens_left_padded))
        shared_out, _ = self.lstm_shared(x)
        shared_out = self.bn_shared(shared_out.permute(0, 2, 1)).permute(0, 2, 1)
        return shared_out

    def forward(self, prefix_tokens_left_padded: torch.Tensor) -> torch.Tensor:
        """Direct (non-autoregressive) suffix prediction: one forward pass
        returns logits for every suffix position at once (batch,
        window_size, num_classes) - used identically at train time
        (teacher forcing is not applicable/needed) and eval time."""
        shared_out = self.encode(prefix_tokens_left_padded)
        act_out, _ = self.lstm_act(shared_out)
        act_out = self.bn_act(act_out.permute(0, 2, 1)).permute(0, 2, 1)
        return self.fc_out_act(act_out)
