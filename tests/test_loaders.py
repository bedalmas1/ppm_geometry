"""Unit tests for src/data/loaders.py using a tiny synthetic CSV — no
dependency on the real (gitignored) Helpdesk download."""
import pandas as pd
import pytest

from data.loaders import load_dataset, load_helpdesk_csv
from data.schema import ACTIVITY, CASE_ID, TIMESTAMP


@pytest.fixture
def toy_csv(tmp_path):
    path = tmp_path / "toy_helpdesk.csv"
    path.write_text(
        "Case ID,Activity,Resource,Complete Timestamp\n"
        "Case 2,Close,R1,2020/01/02 09:00:00.000\n"
        "Case 1,Open,R1,2020/01/01 10:00:00.000\n"
        "Case 1,Close,R2,2020/01/01 11:30:00.500\n"
        "Case 2,Open,R2,2020/01/02 08:00:00.000\n",
        encoding="utf-8",
    )
    return path


def test_helpdesk_csv_maps_to_common_schema(toy_csv):
    df = load_helpdesk_csv(toy_csv)
    assert {CASE_ID, ACTIVITY, TIMESTAMP} <= set(df.columns)
    assert df[TIMESTAMP].dtype.kind == "M"


def test_helpdesk_csv_sorted_by_case_start_then_event_time(toy_csv):
    df = load_helpdesk_csv(toy_csv)
    # Case 1 started before Case 2, so all of case_1's events should precede
    # case_2's, even though the raw CSV interleaves them.
    case_order = list(dict.fromkeys(df[CASE_ID]))
    assert case_order == ["Case 1", "Case 2"]
    # within case_1, Open (10:00) must precede Close (11:30)
    case1 = df[df[CASE_ID] == "Case 1"]
    assert list(case1[ACTIVITY]) == ["Open", "Close"]


def test_load_dataset_dispatches_by_format(toy_csv):
    df = load_dataset(toy_csv, "csv")
    assert len(df) == 4


def test_load_dataset_rejects_unknown_format(toy_csv):
    with pytest.raises(ValueError, match="Unknown format"):
        load_dataset(toy_csv, "parquet")
