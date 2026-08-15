"""Descriptive statistics required by spec §6, computed from a loaded event log.

Every number here is derived from the actual parsed data — never hand-copied
from a paper or landing page (per PLAN.md's cross-cutting reminders).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from data.schema import ACTIVITY, CASE_ID, TIMESTAMP


@dataclass(frozen=True)
class DatasetStats:
    n_cases: int
    n_events: int
    n_activities: int
    mean_trace_length: float
    median_trace_length: float
    n_trace_variants: int
    variant_entropy_bits: float
    temporal_span_start: str
    temporal_span_end: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_stats(df: pd.DataFrame) -> DatasetStats:
    trace_lengths = df.groupby(CASE_ID).size()

    variants = df.groupby(CASE_ID)[ACTIVITY].agg(tuple)
    variant_counts = variants.value_counts()
    variant_probs = variant_counts / variant_counts.sum()
    variant_entropy_bits = float(-(variant_probs * np.log2(variant_probs)).sum())

    return DatasetStats(
        n_cases=int(df[CASE_ID].nunique()),
        n_events=int(len(df)),
        n_activities=int(df[ACTIVITY].nunique()),
        mean_trace_length=float(trace_lengths.mean()),
        median_trace_length=float(trace_lengths.median()),
        n_trace_variants=int(variant_counts.shape[0]),
        variant_entropy_bits=variant_entropy_bits,
        temporal_span_start=str(df[TIMESTAMP].min()),
        temporal_span_end=str(df[TIMESTAMP].max()),
    )
