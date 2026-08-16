"""Adapter: this project's common event-log schema -> B2's expected
full-suffix tensors.

Uses the shared suffix-prefix definition (data.prefixes.make_suffix_prefixes)
- the same one A4 SuTraN and A5 CRTP-LSTM are built on - so B2 is trained on
exactly the same (history, suffix) pairs as every other full-suffix model in
the roster. Written independently of A4's own adapter.py (rather than
importing it, unlike A5's reuse) to keep Family B's provenance unambiguous -
same rationale as model.py's decision not to reuse A4's vendored attention
layers. The two-vocabulary convention (PAD/UNK on the word/input side,
PAD/EOS/UNK on the class/target side, with a fixed +1 offset between them)
mirrors A4/A5's adapter because that structure is a necessary consequence of
the task itself (a closed TRAIN-only vocab, EOS-terminated suffixes), not a
paper-specific trick.

Unlike A4/A5, prefix and suffix sequences use INDEPENDENT padded window
lengths (`get_prefix_window`/`get_suffix_window`) rather than one shared
`window_size` - see model.py's `ControlledTransformerSuffix` docstring for
why that shared-window requirement was specific to SuTraN's own hand-rolled
attention mask-broadcasting, not a general Transformer constraint.
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
    """Word-vocab (encoder/decoder input tokens) and class-vocab
    (decoder output/target classes) share the same underlying activity
    ordering, so converting a predicted class index back into a decoder
    input token is a fixed +1 offset (see model.py's generate()):
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


def get_prefix_window(prefix_df: pd.DataFrame) -> int:
    return int(prefix_df["history"].map(len).max())


def get_suffix_window(prefix_df: pd.DataFrame) -> int:
    return int(prefix_df["suffix"].map(len).max())


def _pad_right(sequences: list[list[int]], maxlen: int, pad_value: int = 0) -> np.ndarray:
    out = np.full((len(sequences), maxlen), pad_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        out[i, :length] = seq[:length]
    return out


def encode_prefixes(prefix_df: pd.DataFrame, vocab: Vocab, prefix_window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (prefix_tokens, prefix_pad_mask): (N, prefix_window) each.
    prefix_pad_mask is True at padded (masked) positions, matching
    nn.TransformerEncoder's/nn.TransformerDecoder's src/memory
    key-padding-mask convention."""
    token_seqs = [[vocab.word_dict.get(a, vocab.word_dict[UNK]) for a in hist] for hist in prefix_df["history"]]
    tokens = _pad_right(token_seqs, prefix_window, pad_value=0)
    lengths = prefix_df["history"].map(len).to_numpy()
    pad_mask = np.arange(prefix_window)[None, :] >= lengths[:, None]
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(pad_mask, dtype=torch.bool)


def encode_training_suffixes(
    prefix_df: pd.DataFrame, vocab: Vocab, suffix_window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (decoder_input_tokens, target_classes) for teacher-forced
    training, both (N, suffix_window):
      decoder_input[0]  = last prefix activity (SOS proxy, word-vocab index)
      decoder_input[1:] = suffix activities excluding the trailing EOS
      target            = suffix (activities + trailing EOS), class-vocab
    Target padding uses class index 0 (PAD), ignored via CrossEntropyLoss's
    ignore_index. No separate decoder padding mask is needed: the causal
    mask already prevents any real position from attending to a later
    (padded) one, and the loss ignores padded target positions directly -
    same convention as A4/A5.
    """
    decoder_inputs, targets = [], []
    for hist, suf in zip(prefix_df["history"], prefix_df["suffix"]):
        sos = vocab.word_dict.get(hist[-1], vocab.word_dict[UNK])
        dec_in = [sos] + [vocab.word_dict.get(a, vocab.word_dict[UNK]) for a in suf[:-1]]
        tgt = [vocab.class_dict.get(a, vocab.class_dict[UNK]) for a in suf]
        decoder_inputs.append(dec_in)
        targets.append(tgt)
    dec_in_arr = _pad_right(decoder_inputs, suffix_window, pad_value=0)
    tgt_arr = _pad_right(targets, suffix_window, pad_value=0)
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
