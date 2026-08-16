"""Adapter: this project's common event-log schema -> MLMME's expected
one-hot prefix/suffix tensors (activity-only scope, remaining-time head and
input dropped - see model.py's docstring for the full scope-decision
rationale).

**LICENSE NOTICE**: see model.py's docstring - GPL-3.0, isolated per this
project's license-isolation policy.

Reuses A4 SuTraN's Vocab/build_vocab/get_window_size/encode_training_suffixes
directly (`models.sutran.adapter`), exactly as A5/A6 already do - MLMME
solves the same activity-only full-suffix task from the same shared
suffix-prefix definition (`data.prefixes.make_suffix_prefixes`).

**A genuine, deliberate difference from A4/A5's adapters**: SuTraN's
`Vocab` has TWO index spaces - a `word_dict` (encoder/decoder INPUT tokens,
offset +2 for PAD/UNK) and a `class_dict` (OUTPUT/target classes) - because
SuTraN needs a learned nn.Embedding table for its input side, and that
table's own PAD/UNK rows are architecturally distinct from the output
softmax's PAD/EOS/UNK rows. MLMME has no embedding layer at all (see
model.py's docstring (a)): every event, on BOTH the encoder's prefix input
side and the decoder's suffix input/output side, is represented directly
as a one-hot vector over ONE flat class space (`preparation.py`'s
`__event_to_one_hot`, which builds one single set of one-hot columns used
identically for prefixes and suffixes). This adapter therefore uses
`vocab.class_dict` (not `word_dict`) for BOTH the prefix and the suffix,
sized `vocab.num_classes` - a deliberate, disclosed adapter difference
matching MLMME's real single-vocabulary representation, not an
inconsistency with A4/A5's own (correctly different, for their own
architecture) two-vocabulary convention.
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


def encode_prefixes_classidx(
    prefix_df: pd.DataFrame, vocab: Vocab, window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (prefix_class_idx, lengths): prefix_class_idx is (N,
    window_size), RIGHT-padded with class index 0 (PAD, matching
    `encode_training_suffixes`'s target-padding convention); `lengths` is
    (N,), the true (unpadded) prefix length per row, needed for
    `pack_padded_sequence` in model.py's `Encoder` (see its docstring for
    why this is used instead of the original repo's own same-length-bucket
    batching trick). One-hot conversion to (N, window_size, num_classes)
    happens lazily via `F.one_hot` in the training script / model.py, not
    stored densely here."""
    from models.sutran.adapter import UNK

    tokens = np.zeros((len(prefix_df), window_size), dtype=np.int64)
    lengths = np.zeros(len(prefix_df), dtype=np.int64)
    for row_idx, hist in enumerate(prefix_df["history"]):
        encoded = [vocab.class_dict.get(a, vocab.class_dict[UNK]) for a in hist]
        length = min(len(encoded), window_size)
        tokens[row_idx, :length] = encoded[:length]
        lengths[row_idx] = length
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
