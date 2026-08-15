"""Canonical event-log schema every loader converges to.

Every dataset in this project, regardless of source format (XES, CSV), is
loaded into a pandas DataFrame with at least these three columns. Downstream
code (splitting, stats, representation extraction) depends only on this
schema, never on a dataset's original column names.
"""

CASE_ID = "case_id"
ACTIVITY = "activity"
TIMESTAMP = "timestamp"

REQUIRED_COLUMNS = (CASE_ID, ACTIVITY, TIMESTAMP)


def validate_schema(df) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Event log is missing required column(s): {missing}")
    if not df[TIMESTAMP].dtype.kind == "M":
        raise ValueError(f"'{TIMESTAMP}' column must be a parsed datetime, got {df[TIMESTAMP].dtype}")
