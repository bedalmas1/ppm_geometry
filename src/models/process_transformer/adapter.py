"""Adapter: this project's common event-log schema -> ProcessTransformer's
expected prefix/next-activity tensors.

Deliberately does NOT reuse the original repo's own data_processing.py -
that would let each model do its own train/test split, which violates this
project's requirement (spec Sec.6) of identical splits across every model.
Instead this reads the already-split parquet produced by
scripts/prepare_dataset.py (src/data/) and reshapes it into exactly the
tensor format processtransformer/models/transformer.py expects.

Two deliberate, documented deviations from the original repo, both in the
direction of correctness rather than fidelity:
  1. Vocabulary (x_word_dict/y_word_dict) is built from the TRAIN split
     only. The original repo builds it from the full dataset (train+test
     combined) before splitting - a train/test leakage shortcut we do not
     replicate. Unseen activities at val/test time map to "[UNK]".
  2. Activity-name normalization (lowercase, spaces -> hyphens) IS
     replicated faithfully, since it's required for the space-joined
     prefix string encoding to round-trip correctly (an activity name
     containing a raw space would corrupt the tokenization).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tensorflow.keras.utils import pad_sequences

from data.prefixes import make_next_activity_prefixes

PAD = "[PAD]"
UNK = "[UNK]"


@dataclass(frozen=True)
class Vocab:
    x_word_dict: dict[str, int]
    y_word_dict: dict[str, int]

    @property
    def vocab_size(self) -> int:
        return len(self.x_word_dict)

    @property
    def num_classes(self) -> int:
        return len(self.y_word_dict)


def build_vocab(train_prefixes: pd.DataFrame) -> Vocab:
    """Build x/y vocabularies from the TRAIN split's prefixes only (see
    module docstring). `train_prefixes` is the output of
    data.prefixes.make_next_activity_prefixes on the train split."""
    activities = sorted({a for hist in train_prefixes["history"] for a in hist} | set(train_prefixes["next_act"]))
    x_word_dict = {PAD: 0, UNK: 1, **{a: i + 2 for i, a in enumerate(activities)}}
    y_word_dict = {a: i for i, a in enumerate(activities)}
    y_word_dict[UNK] = len(y_word_dict)  # reserved class for unseen test/val labels
    return Vocab(x_word_dict, y_word_dict)


# Re-exported so callers only need to import this module for the whole
# ProcessTransformer data pipeline.
make_prefixes = make_next_activity_prefixes


def get_max_case_length(prefix_df: pd.DataFrame) -> int:
    return int(prefix_df["history"].map(len).max())


def encode(prefix_df: pd.DataFrame, vocab: Vocab, max_case_length: int) -> tuple[np.ndarray, np.ndarray]:
    token_x = [
        [vocab.x_word_dict.get(tok, vocab.x_word_dict[UNK]) for tok in hist] for hist in prefix_df["history"]
    ]
    token_x = pad_sequences(token_x, maxlen=max_case_length)
    token_y = np.array(
        [vocab.y_word_dict.get(a, vocab.y_word_dict[UNK]) for a in prefix_df["next_act"]],
        dtype=np.int64,
    )
    return np.asarray(token_x, dtype=np.float32), token_y
