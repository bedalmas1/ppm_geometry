"""Adapter: this project's common event-log schema -> SuTraN's expected
prefix/suffix tensors (activity-only scope, see model.py's docstring).

Uses the shared suffix-prefix definition
(data.prefixes.make_suffix_prefixes) so every full-suffix model in this
project's roster (A4 SuTraN, A5 CRTP-LSTM, B2) is trained on exactly the
same (history, suffix) pairs per case. Vocabulary is built from the TRAIN
split only, same rationale as A1/A2's adapters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from data.prefixes import EOS

PAD = "[PAD]"
UNK = "[UNK]"


@dataclass(frozen=True)
class Vocab:
    """Word-vocab (encoder/decoder input tokens) and class-vocab (decoder
    output/target classes) share the same underlying activity ordering, so
    converting a predicted class index back into a decoder input token is a
    fixed +1 offset (see model.py's generate()):
      word_dict:  PAD=0, UNK=1, activity_i = i+2
      class_dict: PAD=0, activity_i = i+1, EOS=V+1, UNK=V+2
    """

    activities: tuple[str, ...]
    word_dict: dict
    class_dict: dict

    @property
    def vocab_size(self) -> int:
        return len(self.word_dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_dict)

    @property
    def eos_class(self) -> int:
        return self.class_dict[EOS]


def build_vocab(train_prefixes: pd.DataFrame) -> Vocab:
    activities = sorted(
        {a for hist in train_prefixes["history"] for a in hist}
        | {a for suf in train_prefixes["suffix"] for a in suf if a != EOS}
    )
    word_dict = {PAD: 0, UNK: 1, **{a: i + 2 for i, a in enumerate(activities)}}
    class_dict = {PAD: 0, **{a: i + 1 for i, a in enumerate(activities)}}
    class_dict[EOS] = len(activities) + 1
    class_dict[UNK] = len(activities) + 2
    return Vocab(tuple(activities), word_dict, class_dict)


def get_window_size(prefix_df: pd.DataFrame) -> int:
    """A single shared max length ('window_size' in the original repo's own
    terminology) used to pad BOTH prefix and suffix tensors. This is not an
    arbitrary simplification: the original repo does exactly this (its
    README documents one shared `window_size` W for both prefix and suffix
    tensors), and it matters architecturally, not just cosmetically -
    cross-attention's key-padding mask (derived from the encoder/prefix
    sequence) is only broadcastable against the decoder/suffix sequence
    length when both share the same padded length. Using two independent
    max lengths (as an earlier version of this adapter did) breaks that
    broadcast and crashes with a shape-mismatch error - a genuine
    architectural constraint, not just a convention to mirror for fidelity."""
    max_prefix_len = int(prefix_df["history"].map(len).max())
    max_suffix_len = int(prefix_df["suffix"].map(len).max())
    return max(max_prefix_len, max_suffix_len)


def _pad_right(sequences: list[list[int]], maxlen: int, pad_value: int = 0) -> np.ndarray:
    out = np.full((len(sequences), maxlen), pad_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        out[i, :length] = seq[:length]
    return out


def encode_prefixes(prefix_df: pd.DataFrame, vocab: Vocab, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (prefix_tokens, prefix_pad_mask): both (N, window_size).
    prefix_pad_mask is True at padded (masked) positions, matching
    src/models/sutran/layers.py's MultiHeadAttention mask convention.
    `window_size` must be the same shared value passed to
    encode_training_suffixes (see get_window_size's docstring)."""
    token_seqs = [[vocab.word_dict.get(a, vocab.word_dict[UNK]) for a in hist] for hist in prefix_df["history"]]
    tokens = _pad_right(token_seqs, window_size, pad_value=0)
    lengths = prefix_df["history"].map(len).to_numpy()
    pad_mask = np.arange(window_size)[None, :] >= lengths[:, None]
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(pad_mask, dtype=torch.bool)


def encode_training_suffixes(
    prefix_df: pd.DataFrame, vocab: Vocab, window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (decoder_input_tokens, target_classes) for teacher-forced
    training, both (N, window_size) - the same shared window_size passed to
    encode_prefixes (see get_window_size's docstring):
      decoder_input[0]  = last prefix activity (SOS proxy, word-vocab index)
      decoder_input[1:] = suffix activities excluding the trailing EOS
      target            = suffix (activities + trailing EOS), class-vocab
    Target padding uses class index 0 (PAD), ignored via CrossEntropyLoss's
    ignore_index.
    """
    decoder_inputs, targets = [], []
    for hist, suf in zip(prefix_df["history"], prefix_df["suffix"]):
        sos = vocab.word_dict.get(hist[-1], vocab.word_dict[UNK])
        dec_in = [sos] + [vocab.word_dict.get(a, vocab.word_dict[UNK]) for a in suf[:-1]]
        tgt = [vocab.class_dict.get(a, vocab.class_dict[UNK]) for a in suf]
        decoder_inputs.append(dec_in)
        targets.append(tgt)
    dec_in_arr = _pad_right(decoder_inputs, window_size, pad_value=0)
    tgt_arr = _pad_right(targets, window_size, pad_value=0)
    return torch.tensor(dec_in_arr, dtype=torch.long), torch.tensor(tgt_arr, dtype=torch.long)


def sos_tokens(prefix_df: pd.DataFrame, vocab: Vocab) -> torch.Tensor:
    """The decoder's start-of-sequence proxy for each instance (last prefix
    activity, word-vocab index) - used at inference (model.generate())."""
    sos = [vocab.word_dict.get(hist[-1], vocab.word_dict[UNK]) for hist in prefix_df["history"]]
    return torch.tensor(sos, dtype=torch.long)


def build_reverse_class_dict(vocab: Vocab) -> dict[int, str]:
    """class index -> activity string (or EOS/UNK/PAD). Build once and
    reuse across an evaluation loop rather than reconstructing per lookup."""
    return {idx: name for name, idx in vocab.class_dict.items()}
