"""Format-specific loaders converging to the common schema in schema.py.

Each loader's only job is: read a raw file, produce a DataFrame with
case_id/activity/timestamp (+ any pass-through attributes), sorted by
(case_id start time, timestamp). Splitting, stats, and everything else is
format-agnostic and lives elsewhere.
"""
from pathlib import Path

import pandas as pd

from data.schema import ACTIVITY, CASE_ID, TIMESTAMP, validate_schema


def load_xes(path: str | Path) -> pd.DataFrame:
    """Load an XES (or .xes.gz) event log into the common schema via pm4py."""
    import pm4py

    df = pm4py.read_xes(str(path))
    df = df.rename(
        columns={
            "case:concept:name": CASE_ID,
            "concept:name": ACTIVITY,
            "time:timestamp": TIMESTAMP,
        }
    )
    df[CASE_ID] = df[CASE_ID].astype(str)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True)
    df = _sort_by_case_start_then_time(df)
    validate_schema(df)
    return df


def load_helpdesk_csv(path: str | Path) -> pd.DataFrame:
    """Load the Helpdesk (Italian company) CSV into the common schema.

    Only non-XES source in the dataset roster — see
    configs/datasets/helpdesk.yaml. Timestamp format confirmed by inspecting
    the raw file: "YYYY/MM/DD HH:MM:SS.ffffff".
    """
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Case ID": CASE_ID,
            "Activity": ACTIVITY,
            "Complete Timestamp": TIMESTAMP,
        }
    )
    df[CASE_ID] = df[CASE_ID].astype(str)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], format="%Y/%m/%d %H:%M:%S.%f")
    df = _sort_by_case_start_then_time(df)
    validate_schema(df)
    return df


def _sort_by_case_start_then_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort cases by their first event's timestamp, events within a case by time.

    This ordering is what splits.py's time-based split relies on to prevent
    temporal leakage (spec §6) — do not reorder rows after this point without
    re-deriving splits.
    """
    case_start = df.groupby(CASE_ID)[TIMESTAMP].transform("min")
    df = df.assign(_case_start=case_start).sort_values(
        ["_case_start", CASE_ID, TIMESTAMP], kind="stable"
    )
    return df.drop(columns="_case_start").reset_index(drop=True)


FORMAT_LOADERS = {
    "xes": load_xes,
    "xes_gz": load_xes,
    "csv": load_helpdesk_csv,
}


def _apply_date_filter(df: pd.DataFrame, date_filter: dict | None) -> pd.DataFrame:
    """Drop individual *events* (not whole cases) whose timestamp falls
    outside [start, end) - BPIC19's only consumer so far (STATUS.md: its
    computed temporal span starts 1948-01-26, a known placeholder-timestamp
    artifact on a handful of events, not a real 72-year process).

    Row-level rather than case-level: removes the corrupted event record
    while keeping the rest of that case's otherwise-valid events. Cutoffs
    match SuTraN's own create_BPIC19_data.py (start_date="2018-01",
    end_date="2019-02"), found while integrating A4 (STATUS.md decision
    log); applying them at the row level (vs. dropping whole cases whose
    span touches the artifact) is this project's own disclosed choice, not
    independently re-verified against SuTraN's exact filtering granularity.
    """
    if date_filter is None:
        return df
    start = pd.Timestamp(date_filter["start"], tz="UTC")
    end = pd.Timestamp(date_filter["end"], tz="UTC")
    mask = (df[TIMESTAMP] >= start) & (df[TIMESTAMP] < end)
    return df.loc[mask].reset_index(drop=True)


def load_dataset(raw_path: str | Path, fmt: str, date_filter: dict | None = None) -> pd.DataFrame:
    """Dispatch to the right loader for a dataset config's declared `format`,
    then optionally apply a row-level timestamp filter (`date_filter`,
    e.g. configs/datasets/bpic19.yaml's `date_filter: {start, end}`) - a
    no-op for every dataset that doesn't declare one."""
    try:
        loader = FORMAT_LOADERS[fmt]
    except KeyError:
        raise ValueError(f"Unknown format '{fmt}'. Known: {list(FORMAT_LOADERS)}") from None
    df = loader(raw_path)
    return _apply_date_filter(df, date_filter)
