"""Adapter: this project's common event-log schema -> B1's expected
next-event tensors.

Uses the shared next-event prefix definition
(data.prefixes.make_next_activity_prefixes) - the same one A1/A2 are built
on - so B1 is trained on exactly the same (history, next_act) pairs as
every other next-event model in the roster. Vocabulary is built from the
TRAIN split only, same rationale as every other adapter in this project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from data.prefixes import make_next_activity_prefixes

PAD = "[PAD]"
UNK = "[UNK]"

# Re-exported so callers only need to import this module for the whole
# B1 data pipeline (same pattern as process_transformer/adapter.py).
make_prefixes = make_next_activity_prefixes


@dataclass(frozen=True)
class Vocab:
    word_dict: dict
    class_dict: dict

    @property
    def vocab_size(self) -> int:
        return len(self.word_dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_dict)


def build_vocab(train_prefixes: pd.DataFrame) -> Vocab:
    """Build the input (word) and target (class) vocabularies from the
    TRAIN split's prefixes only. `train_prefixes` is the output of
    make_next_activity_prefixes on the train split. Unseen activities at
    val/test time map to [UNK] on both sides."""
    activities = sorted({a for hist in train_prefixes["history"] for a in hist} | set(train_prefixes["next_act"]))
    word_dict = {PAD: 0, UNK: 1, **{a: i + 2 for i, a in enumerate(activities)}}
    class_dict = {a: i for i, a in enumerate(activities)}
    class_dict[UNK] = len(class_dict)  # reserved class for unseen test/val labels
    return Vocab(word_dict, class_dict)


def get_max_case_length(prefix_df: pd.DataFrame) -> int:
    return int(prefix_df["history"].map(len).max())


def encode(
    prefix_df: pd.DataFrame, vocab: Vocab, max_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (tokens, pad_mask, lengths, labels):
      tokens    - (N, max_len) right-padded word-vocab ids
      pad_mask  - (N, max_len) bool, True at padded positions (matches
                  nn.TransformerEncoder's src_key_padding_mask convention)
      lengths   - (N,) real (non-padded) prefix length per example
      labels    - (N,) class-vocab next-activity target
    """
    n = len(prefix_df)
    tokens = np.zeros((n, max_len), dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    for i, hist in enumerate(prefix_df["history"]):
        ids = [vocab.word_dict.get(a, vocab.word_dict[UNK]) for a in hist][:max_len]
        tokens[i, : len(ids)] = ids
        lengths[i] = len(ids)
    pad_mask = np.arange(max_len)[None, :] >= lengths[:, None]
    labels = np.array(
        [vocab.class_dict.get(a, vocab.class_dict[UNK]) for a in prefix_df["next_act"]],
        dtype=np.int64,
    )
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(pad_mask, dtype=torch.bool),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
