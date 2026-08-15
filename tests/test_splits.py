"""Unit tests for src/data/splits.py using synthetic event logs.

No dependency on the real (gitignored, locally-downloaded) datasets — these
tests must pass on a clean checkout.
"""
import pandas as pd
import pytest

from data.splits import apply_split, compute_split
from data.schema import CASE_ID, ACTIVITY, TIMESTAMP


def make_log(n_cases: int) -> pd.DataFrame:
    """Cases 'case_0'..'case_{n-1}', each starting one day after the last,
    each with 2 events. Already in the start-time-sorted order splits.py
    expects (mirrors what loaders._sort_by_case_start_then_time produces)."""
    rows = []
    for i in range(n_cases):
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
        rows.append({CASE_ID: f"case_{i}", ACTIVITY: "A", TIMESTAMP: start})
        rows.append({CASE_ID: f"case_{i}", ACTIVITY: "B", TIMESTAMP: start + pd.Timedelta(hours=1)})
    return pd.DataFrame(rows)


def test_split_sizes_match_fractions():
    df = make_log(100)
    split = compute_split(df, train_frac=0.64, val_frac=0.16, test_frac=0.20)
    assert len(split.train_cases) == 64
    assert len(split.val_cases) == 16
    assert len(split.test_cases) == 20


def test_split_is_time_ordered_not_random():
    df = make_log(10)
    split = compute_split(df, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    # earliest-starting cases go to train, latest to test — never shuffled
    assert split.train_cases == ("case_0", "case_1", "case_2", "case_3", "case_4", "case_5")
    assert split.val_cases == ("case_6", "case_7")
    assert split.test_cases == ("case_8", "case_9")


def test_splits_are_disjoint_and_cover_all_cases():
    df = make_log(37)  # deliberately not evenly divisible
    split = compute_split(df)
    all_cases = set(split.train_cases) | set(split.val_cases) | set(split.test_cases)
    assert all_cases == set(df[CASE_ID].unique())
    assert not (set(split.train_cases) & set(split.val_cases))
    assert not (set(split.val_cases) & set(split.test_cases))
    assert not (set(split.train_cases) & set(split.test_cases))


def test_split_hash_is_deterministic():
    df = make_log(20)
    split_a = compute_split(df)
    split_b = compute_split(df)
    assert split_a.split_hash == split_b.split_hash


def test_split_hash_changes_when_data_changes():
    split_a = compute_split(make_log(20))
    split_b = compute_split(make_log(21))
    assert split_a.split_hash != split_b.split_hash


def test_fractions_must_sum_to_one():
    df = make_log(10)
    with pytest.raises(ValueError):
        compute_split(df, train_frac=0.5, val_frac=0.5, test_frac=0.5)


def test_apply_split_partitions_events_correctly():
    df = make_log(10)
    split = compute_split(df, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    parts = apply_split(df, split)

    assert set(parts["train"][CASE_ID].unique()) == set(split.train_cases)
    assert set(parts["val"][CASE_ID].unique()) == set(split.val_cases)
    assert set(parts["test"][CASE_ID].unique()) == set(split.test_cases)
    # every event accounted for exactly once
    assert len(parts["train"]) + len(parts["val"]) + len(parts["test"]) == len(df)
