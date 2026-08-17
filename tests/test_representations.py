"""Unit tests for src/representations/ (Phase 4: representation extraction).

No dependency on the real (gitignored) datasets or on any trained
checkpoint - common_fields/cache tests use synthetic data; the smoke test
instantiates a tiny, freshly-initialized (untrained) model to check the
extraction-shape logic, not any learned behavior.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.prefixes import make_next_activity_prefixes, make_suffix_prefixes
from data.schema import ACTIVITY, CASE_ID, TIMESTAMP
from representations.cache import build_provenance, is_stale
from representations.common_fields import compute_common_fields


def make_log() -> pd.DataFrame:
    """One case, 3 events: A (t=0h), B (t=+1h), C (t=+3h)."""
    start = pd.Timestamp("2020-01-01")
    return pd.DataFrame(
        [
            {CASE_ID: "case_0", ACTIVITY: "A", TIMESTAMP: start},
            {CASE_ID: "case_0", ACTIVITY: "B", TIMESTAMP: start + pd.Timedelta(hours=1)},
            {CASE_ID: "case_0", ACTIVITY: "C", TIMESTAMP: start + pd.Timedelta(hours=3)},
        ]
    )


def test_common_fields_next_event_prefixes():
    df = make_log()
    prefixes = make_next_activity_prefixes(df)  # k=0 (hist=[a]), k=1 (hist=[a,b]) - no terminal row
    out = compute_common_fields(df, prefixes).set_index("k")

    assert len(out) == 2
    assert out.loc[0, "prefix_length"] == 1
    assert out.loc[0, "case_length"] == 3
    assert out.loc[0, "true_event"] == "b"
    assert out.loc[0, "remaining_time_seconds"] == pytest.approx(3 * 3600)
    assert out.loc[0, "outcome"] is None

    assert out.loc[1, "prefix_length"] == 2
    assert out.loc[1, "true_event"] == "c"
    assert out.loc[1, "remaining_time_seconds"] == pytest.approx(2 * 3600)


def test_common_fields_suffix_prefixes_include_terminal_row():
    df = make_log()
    prefixes = make_suffix_prefixes(df)  # k=0,1,2 - includes the complete-case row
    out = compute_common_fields(df, prefixes).set_index("k")

    assert len(out) == 3  # one more row than the next-event case: the terminal prefix
    assert out.loc[0, "true_event"] == "b"
    assert out.loc[1, "true_event"] == "c"
    assert pd.isna(out.loc[2, "true_event"])  # terminal row: suffix is [EOS] only, no next event
    assert out.loc[2, "remaining_time_seconds"] == pytest.approx(0.0)
    assert out.loc[2, "case_length"] == 3


def _provenance_factory(tmp_path, checkpoint, code):
    def provenance(**overrides) -> dict:
        kwargs = dict(
            dataset="helpdesk",
            model="dummy",
            checkpoint_path=checkpoint,
            dataset_split_hash="split-hash",
            extraction_code_paths=[code],
            z_dim=8,
            n_rows=100,
            extracted_at_utc="2026-01-01T00:00:00+00:00",
            git_commit="deadbeef",
            software_versions={"torch": "0.0.0"},
            output_path=tmp_path / "embeddings_test.parquet",
        )
        kwargs.update(overrides)
        return build_provenance(**kwargs)

    return provenance


def test_is_stale_true_when_no_existing_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"v1")
    code = tmp_path / "extract_dummy.py"
    code.write_text("# v1\n", encoding="utf-8")
    provenance = _provenance_factory(tmp_path, checkpoint, code)()
    assert is_stale(None, provenance) is True


def test_is_stale_false_when_nothing_changed(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"v1")
    code = tmp_path / "extract_dummy.py"
    code.write_text("# v1\n", encoding="utf-8")
    provenance = _provenance_factory(tmp_path, checkpoint, code)()
    assert is_stale(provenance, provenance) is False


def test_is_stale_true_when_checkpoint_bytes_change(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"v1")
    code = tmp_path / "extract_dummy.py"
    code.write_text("# v1\n", encoding="utf-8")
    make_provenance = _provenance_factory(tmp_path, checkpoint, code)

    old = make_provenance()
    checkpoint.write_bytes(b"v2-different-weights")
    new = make_provenance()
    assert is_stale(old, new) is True


def test_is_stale_true_when_split_hash_changes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"v1")
    code = tmp_path / "extract_dummy.py"
    code.write_text("# v1\n", encoding="utf-8")
    make_provenance = _provenance_factory(tmp_path, checkpoint, code)

    old = make_provenance()
    new = make_provenance(dataset_split_hash="a-different-split-hash")
    assert is_stale(old, new) is True


def test_is_stale_true_when_extraction_code_changes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"v1")
    code = tmp_path / "extract_dummy.py"
    code.write_text("# v1\n", encoding="utf-8")
    make_provenance = _provenance_factory(tmp_path, checkpoint, code)

    old = make_provenance()
    code.write_text("# v2 - logic changed\n", encoding="utf-8")
    new = make_provenance()
    assert is_stale(old, new) is True


def test_smoke_controlled_transformer_next_zt_shape():
    """One representative per-model smoke test (Family B, simplest
    architecture): a freshly-initialized (untrained) model still produces
    z_t of the documented shape/one-vector-per-prefix over a tiny synthetic
    batch - catches extraction-shape bugs without needing a real checkpoint
    or the full roster's other 8 architectures/frameworks."""
    torch = pytest.importorskip("torch")
    from models.controlled_transformer.model import ControlledTransformerNextEvent

    vocab_size, num_classes, d_model = 6, 4, 8
    model = ControlledTransformerNextEvent(
        vocab_size=vocab_size, num_classes=num_classes, d_model=d_model, num_heads=2, num_layers=1, d_ff=16
    )
    model.eval()

    tokens = torch.tensor([[2, 3, 0], [2, 3, 4]], dtype=torch.long)
    pad_mask = torch.tensor([[False, False, True], [False, False, False]], dtype=torch.bool)
    lengths = torch.tensor([2, 3], dtype=torch.long)

    with torch.no_grad():
        z_t = model.encode_zt(tokens, pad_mask, lengths)
        logits = model.classifier(z_t)

    assert z_t.shape == (2, d_model)
    assert logits.shape == (2, num_classes)
