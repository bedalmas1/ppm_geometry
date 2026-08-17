"""Model-agnostic per-prefix fields shared by every extraction script.

None of these columns depend on any model's architecture or checkpoint -
they are derivable purely from the raw event log and the (history, k, ...)
prefix table already produced by data.prefixes.make_next_activity_prefixes
or make_suffix_prefixes. Computed once per (dataset, model) run and then
left-joined (on case_id, k) against that model's own z_t/prediction columns.
"""
from __future__ import annotations

import pandas as pd

from data.prefixes import EOS
from data.schema import CASE_ID, TIMESTAMP


def _case_event_timestamps(raw_df: pd.DataFrame) -> dict:
    """case_id -> list of that case's event timestamps, in original
    (already case-start-then-time sorted, see data.loaders) row order."""
    return raw_df.groupby(CASE_ID)[TIMESTAMP].apply(list).to_dict()


def compute_common_fields(raw_df: pd.DataFrame, prefix_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (case_id, k) with: prefix_length (=k+1), case_length,
    true_event (the actual next activity; None for a full-suffix model's
    terminal/complete-case row, which has no next event),
    remaining_time_seconds (case's last event timestamp minus this
    prefix's last observed event timestamp), and outcome.

    `raw_df` is the split's raw event-log dataframe (e.g. the test split's
    output of data.splits.apply_split). `prefix_df` is that same split's
    output of make_next_activity_prefixes (has a `next_act` column) or
    make_suffix_prefixes (has a `suffix` column) - either is accepted,
    detected by which column is present.

    `outcome` is always None: this project's canonical schema
    (case_id/activity/timestamp only) has no outcome label, and Phase 1
    deliberately dropped outcome-prediction as a training objective for
    every model in the roster (STATUS.md's decision log) - left null here
    rather than guessed, per that same deliberate deferral.
    """
    ts_by_case = _case_event_timestamps(raw_df)
    is_suffix = "suffix" in prefix_df.columns

    rows = []
    for row in prefix_df.itertuples(index=False):
        case_id = row.case_id
        k = int(row.k)
        timestamps = ts_by_case[case_id]
        case_length = len(timestamps)
        remaining_time_seconds = (timestamps[-1] - timestamps[k]).total_seconds()

        if is_suffix:
            suffix = row.suffix
            true_event = suffix[0] if suffix and suffix[0] != EOS else None
        else:
            true_event = row.next_act

        rows.append(
            {
                "case_id": case_id,
                "k": k,
                "prefix_length": k + 1,
                "case_length": case_length,
                "true_event": true_event,
                "remaining_time_seconds": float(remaining_time_seconds),
                "outcome": None,
            }
        )
    return pd.DataFrame(rows)
