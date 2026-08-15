"""Unit tests for src/data/stats.py against a hand-worked synthetic log
where every expected value is analytically known (same philosophy the
spec requires for the geometry metrics in Phase 5)."""
import math

import pandas as pd
import pytest

from data.schema import ACTIVITY, CASE_ID, TIMESTAMP
from data.stats import compute_stats


@pytest.fixture
def toy_log() -> pd.DataFrame:
    # case_1 and case_2 share the variant (A, B); case_3 is (A, C).
    rows = [
        ("case_1", "A", "2020-01-01T00:00:00"),
        ("case_1", "B", "2020-01-01T01:00:00"),
        ("case_2", "A", "2020-01-02T00:00:00"),
        ("case_2", "B", "2020-01-02T01:00:00"),
        ("case_3", "A", "2020-01-03T00:00:00"),
        ("case_3", "C", "2020-01-03T02:00:00"),
    ]
    df = pd.DataFrame(rows, columns=[CASE_ID, ACTIVITY, TIMESTAMP])
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    return df


def test_counts(toy_log):
    stats = compute_stats(toy_log)
    assert stats.n_cases == 3
    assert stats.n_events == 6
    assert stats.n_activities == 3  # A, B, C


def test_trace_length(toy_log):
    stats = compute_stats(toy_log)
    # every trace has exactly 2 events
    assert stats.mean_trace_length == pytest.approx(2.0)
    assert stats.median_trace_length == pytest.approx(2.0)


def test_variant_count(toy_log):
    stats = compute_stats(toy_log)
    # variants: (A,B) x2, (A,C) x1 -> 2 distinct variants
    assert stats.n_trace_variants == 2


def test_variant_entropy_matches_hand_calculation(toy_log):
    stats = compute_stats(toy_log)
    p = [2 / 3, 1 / 3]
    expected_entropy = -sum(pi * math.log2(pi) for pi in p)
    assert stats.variant_entropy_bits == pytest.approx(expected_entropy, abs=1e-9)


def test_temporal_span(toy_log):
    stats = compute_stats(toy_log)
    assert stats.temporal_span_start == str(pd.Timestamp("2020-01-01T00:00:00"))
    assert stats.temporal_span_end == str(pd.Timestamp("2020-01-03T02:00:00"))


def test_collapsed_case_entropy_is_zero():
    """All cases sharing one variant -> zero entropy (no diversity)."""
    rows = [
        ("c1", "A", "2020-01-01T00:00:00"),
        ("c1", "B", "2020-01-01T01:00:00"),
        ("c2", "A", "2020-01-02T00:00:00"),
        ("c2", "B", "2020-01-02T01:00:00"),
    ]
    df = pd.DataFrame(rows, columns=[CASE_ID, ACTIVITY, TIMESTAMP])
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    stats = compute_stats(df)
    assert stats.n_trace_variants == 1
    assert stats.variant_entropy_bits == pytest.approx(0.0)
