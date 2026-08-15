"""Adapter: this project's common event-log schema -> CRTP-LSTM's expected
prefix/suffix tensors.

Reuses A4 SuTraN's Vocab/build_vocab/encode_training_suffixes directly
(models.sutran.adapter) - both models solve the exact same activity-only
full-suffix task from the exact same shared suffix-prefix definition
(data.prefixes.make_suffix_prefixes), so there is no reason to duplicate
vocabulary or target-encoding logic. The only genuinely CRTP-LSTM-specific
piece is left-padding the prefix input (see model.py's docstring for why).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models.sutran.adapter import (  # noqa: F401
    Vocab,
    build_reverse_class_dict,
    build_vocab,
    encode_training_suffixes,
    get_window_size,
)


def encode_prefixes_left_padded(prefix_df: pd.DataFrame, vocab: Vocab, window_size: int) -> torch.Tensor:
    """Returns prefix_tokens, shape (N, window_size), LEFT-padded: the real
    prefix activities occupy the LAST len(history) positions, padding
    (index 0) fills the rest at the start. See model.py's docstring for
    why this specific padding side is architecturally required."""
    from models.sutran.adapter import UNK

    tokens = np.zeros((len(prefix_df), window_size), dtype=np.int64)
    for row_idx, hist in enumerate(prefix_df["history"]):
        encoded = [vocab.word_dict.get(a, vocab.word_dict[UNK]) for a in hist]
        length = min(len(encoded), window_size)
        tokens[row_idx, window_size - length :] = encoded[-length:]
    return torch.tensor(tokens, dtype=torch.long)
