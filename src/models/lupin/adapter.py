"""Adapter: this project's common event-log schema -> LUPIN's expected
text-story input + per-position suffix classification targets.

**LICENSE NOTICE**: see model.py's docstring - this module is derived from
https://github.com/vinspdb/LUPIN (CC BY-NC-SA 4.0), must stay isolated from
the rest of this project's permissively-licensed code, non-commercial use
only.

Reuses A4 SuTraN's Vocab/build_vocab/get_window_size/encode_training_suffixes
directly (`models.sutran.adapter`), exactly as A5 CRTP-LSTM already does -
LUPIN solves the same activity-only full-suffix task from the same shared
suffix-prefix definition (`data.prefixes.make_suffix_prefixes`), so the
OUTPUT side (closed activity class vocabulary, right-padded per-position
suffix targets) needs no LUPIN-specific code at all. The only genuinely
LUPIN-specific pieces are on the INPUT side: turning a `history` (list of
activity names) into a natural-language text story, and tokenizing that text
with the pretrained BERT checkpoint's OWN subword tokenizer (never this
project's closed activity vocabulary - see model.py's docstring for why
these are two deliberately different vocabularies).
"""
from __future__ import annotations

import pandas as pd
import torch

from models.sutran.adapter import (  # noqa: F401
    Vocab,
    build_reverse_class_dict,
    build_vocab,
    encode_training_suffixes,
    get_window_size,
)


def build_prefix_text(history: list[str]) -> str:
    """One narrative sentence per event, activity-only.

    Scoped-down replacement for the original repo's
    `preprocessing/log_to_history.py::__gen_prefix_history`, which renders a
    Jinja2 `event_template` per event weaving in activity + resource +
    elapsed-time + several dataset-specific attributes (see
    `utility/log_config.py`), then appends one `trace_template` sentence for
    case-level attributes. This project's common schema carries only
    activity (see model.py's docstring for the full scope-decision
    rationale), so every clause here mentions only the activity name -
    dropped attributes are not replaced with placeholders, they are simply
    absent, exactly matching the project-wide activity-only convention A1-A5
    already established.
    """
    parts = []
    for i, act in enumerate(history):
        if i == 0:
            parts.append(f"Activity {act} was performed.")
        else:
            parts.append(f"Then activity {act} was performed.")
    return " ".join(parts)


def build_prefix_texts(prefix_df: pd.DataFrame) -> list[str]:
    """One text story per (history, suffix) row, in row order - see
    `build_prefix_text`."""
    return [build_prefix_text(hist) for hist in prefix_df["history"]]


def max_encoded_length(texts: list[str], tokenizer) -> int:
    """Longest BERT-subword token count across `texts` (no special-token
    padding added by the caller here - `encode_prefix_texts` adds
    [CLS]/[SEP] via `add_special_tokens=True`, so callers should add a small
    margin, e.g. +2, when using this as `max_length`).

    Used to pick this project's own `max_token_length` for LUPIN instead of
    the original repo's own `MAX_LEN=512` (see model.py's docstring): this
    project's activity-only text stories are far shorter than the
    original's multi-attribute ones, so reusing its 512-token budget would
    mostly add wasted padding/compute. The caller (`experiments/
    train_lupin.py`) additionally CAPS this value at a smaller number
    (`configs/models/lupin.yaml`'s `max_token_length_cap`) for measured
    CPU-compute reasons - that cap does truncate the longest tail of
    prefixes (see model.py's docstring for the disclosed, non-hidden
    accuracy trade-off). Computed from the TRAIN split only, same rationale
    as vocabulary construction (this project's leakage-safety convention
    applied to a new kind of "vocabulary" - the tokenizer's length budget).
    """
    return max(len(tokenizer.tokenize(t)) for t in texts) if texts else 0


def encode_prefix_texts(
    texts: list[str], tokenizer, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-tokenize `texts` with the pretrained BERT tokenizer (WordPiece
    subwords, NOT this project's closed activity vocabulary - see model.py's
    docstring). Returns (input_ids, attention_mask), both (N, max_length).
    Right-padded/truncated to `max_length` (BERT's own convention -
    `truncation_side='left'` is set on the tokenizer itself at construction
    time in `experiments/train_lupin.py`, matching the original repo's own
    `main.py` choice: if a prefix's text ever needs truncating, drop the
    OLDEST events, not the most recent ones)."""
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]
