"""Model-agnostic next-activity prefix generation.

Every next-event model in this project's roster (A1 ProcessTransformer, A2
GenerativeLSTM, later B1) is trained on the same definition of "a prefix
example": for a case with activities [a0, a1, ..., a_{T-1}], one training
example per i in 0..T-2 of (history=[a0..ai], k=i, next_act=a_{i+1}).
Sharing this single definition across every model - rather than letting each
model's own adapter re-derive prefixes its own way - is what makes Phase 4's
z_t extraction (one representation per prefix length, per model) directly
comparable across the roster.

Each model's own adapter is responsible only for encoding this generic table
into whatever tensor shape its architecture expects (e.g. a space-joined
token string for ProcessTransformer's word-dict scheme, or a plain integer
sequence for an Embedding-layer-based model).
"""
from __future__ import annotations

import pandas as pd

from data.schema import ACTIVITY, CASE_ID


def normalize_activity(name: str) -> str:
    """Lowercase, spaces->hyphens - required wherever an activity name is
    later joined into a single space-separated string (e.g. ProcessTransformer's
    prefix encoding); harmless whitespace-only normalization otherwise."""
    return str(name).lower().replace(" ", "-")


def make_next_activity_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (history, next-activity) pair. `history` is a list of
    normalized activity names (not yet joined/tokenized - that's each
    model's own adapter's job)."""
    rows = []
    for case_id, group in df.groupby(CASE_ID, sort=False):
        acts = [normalize_activity(a) for a in group[ACTIVITY].tolist()]
        for i in range(len(acts) - 1):
            rows.append(
                {
                    "case_id": case_id,
                    "history": acts[: i + 1],
                    "k": i,
                    "next_act": acts[i + 1],
                }
            )
    return pd.DataFrame(rows, columns=["case_id", "history", "k", "next_act"])
