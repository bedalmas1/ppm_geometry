"""Adapter: this project's common event-log schema -> GenerativeLSTM's
expected next-activity tensors.

Uses the shared prefix definition (data.prefixes.make_next_activity_prefixes)
so A1 and A2 are trained on exactly the same (history, next-activity) pairs
per case - see that module's docstring. Vocabulary is built from the TRAIN
split only (same rationale as ProcessTransformer's adapter: the original
repo's own embedding-pretraining step is fit on data outside this project's
scope, see model.py's docstring for the full list of scope decisions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tensorflow.keras.utils import pad_sequences

PAD = "[PAD]"
UNK = "[UNK]"


@dataclass(frozen=True)
class Vocab:
    word_dict: dict[str, int]  # input vocab: PAD, UNK, then activities
    class_dict: dict[str, int]  # output vocab: activities, then a reserved UNK class

    @property
    def vocab_size(self) -> int:
        return len(self.word_dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_dict)


def build_vocab(train_prefixes: pd.DataFrame) -> Vocab:
    activities = sorted({a for hist in train_prefixes["history"] for a in hist} | set(train_prefixes["next_act"]))
    word_dict = {PAD: 0, UNK: 1, **{a: i + 2 for i, a in enumerate(activities)}}
    class_dict = {a: i for i, a in enumerate(activities)}
    class_dict[UNK] = len(class_dict)  # reserved output class for unseen val/test labels
    return Vocab(word_dict, class_dict)


def get_max_case_length(prefix_df: pd.DataFrame) -> int:
    return int(prefix_df["history"].map(len).max())


def encode(prefix_df: pd.DataFrame, vocab: Vocab, max_case_length: int) -> tuple[np.ndarray, np.ndarray]:
    token_x = [[vocab.word_dict.get(tok, vocab.word_dict[UNK]) for tok in hist] for hist in prefix_df["history"]]
    token_x = pad_sequences(token_x, maxlen=max_case_length)
    token_y = np.array(
        [vocab.class_dict.get(a, vocab.class_dict[UNK]) for a in prefix_df["next_act"]], dtype=np.int64
    )
    return np.asarray(token_x, dtype=np.float32), token_y
