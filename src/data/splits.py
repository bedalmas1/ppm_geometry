"""Time-based, leakage-safe train/val/test splitting.

Cases are split by their position in start-time order, never randomly —
a random split would let the model implicitly learn from process behavior
in cases that started later than ones assigned to a later split, which is
exactly the temporal leakage spec §6 warns against.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from data.schema import CASE_ID, TIMESTAMP


@dataclass(frozen=True)
class SplitAssignment:
    train_cases: tuple[str, ...]
    val_cases: tuple[str, ...]
    test_cases: tuple[str, ...]
    split_hash: str


def compute_split(
    df: pd.DataFrame,
    train_frac: float = 0.64,
    val_frac: float = 0.16,
    test_frac: float = 0.20,
) -> SplitAssignment:
    """Assign each case to train/val/test by its position in start-time order.

    `df` must already be sorted by case start time (see
    loaders._sort_by_case_start_then_time) — this function trusts that
    ordering rather than re-deriving it, so case order in `df` IS the split
    order.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError(
            f"train/val/test fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
        )

    ordered_cases = pd.unique(df[CASE_ID])
    n = len(ordered_cases)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_cases = tuple(ordered_cases[:n_train])
    val_cases = tuple(ordered_cases[n_train : n_train + n_val])
    test_cases = tuple(ordered_cases[n_train + n_val :])

    split_hash = _hash_split(train_cases, val_cases, test_cases)
    return SplitAssignment(train_cases, val_cases, test_cases, split_hash)


def _hash_split(train: tuple, val: tuple, test: tuple) -> str:
    """Content hash of the exact case assignment, for provenance (spec §21).

    Any change to the raw data, loader, or split fractions that shifts which
    cases land in which split will change this hash — a cheap way to detect
    a stale cached split.
    """
    h = hashlib.sha256()
    for group in (train, val, test):
        h.update(b"|".join(c.encode("utf-8") for c in group))
        h.update(b"--")
    return h.hexdigest()


def apply_split(df: pd.DataFrame, split: SplitAssignment) -> dict[str, pd.DataFrame]:
    """Slice a loaded event log into train/val/test DataFrames per `split`."""
    train_set, val_set, test_set = set(split.train_cases), set(split.val_cases), set(split.test_cases)
    return {
        "train": df[df[CASE_ID].isin(train_set)].reset_index(drop=True),
        "val": df[df[CASE_ID].isin(val_set)].reset_index(drop=True),
        "test": df[df[CASE_ID].isin(test_set)].reset_index(drop=True),
    }
