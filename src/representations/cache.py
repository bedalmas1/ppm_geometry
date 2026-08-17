"""Embedding-cache provenance/staleness (PLAN.md's Phase 4 requirement:
"Version/hash the embedding cache against (model checkpoint, dataset split,
extraction code) so stale caches are detectable").

Each `experiments/extract_<model>.py` writes one `embeddings_manifest.json`
per (dataset, model) containing the dict `build_provenance` returns, and
checks `is_stale` against any existing one before recomputing - so re-running
an extraction script is a cheap no-op once its cache is fresh.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_files(paths) -> str:
    """Order-independent content hash over several files (e.g. this
    project's shared src/representations/*.py modules plus a specific
    model's own extract_<model>.py) - sorted by path so the result doesn't
    depend on call-site ordering."""
    h = hashlib.sha256()
    for p in sorted(str(p) for p in paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def build_provenance(
    *,
    dataset: str,
    model: str,
    checkpoint_path,
    dataset_split_hash: str,
    extraction_code_paths,
    z_dim: int,
    n_rows: int,
    extracted_at_utc: str,
    git_commit: str,
    software_versions: dict,
    output_path,
) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "extracted_at_utc": extracted_at_utc,
        "git_commit": git_commit,
        "software_versions": software_versions,
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_split_hash": dataset_split_hash,
        "extraction_code_sha256": sha256_files(extraction_code_paths),
        "z_dim": z_dim,
        "n_rows": n_rows,
        "entropy_log_base": "e",
        "output_path": str(output_path),
    }


# Fields that, if any differ from a previous run, mean the cached
# embeddings file no longer reflects the current checkpoint/split/code and
# must be regenerated.
_STALENESS_KEYS = (
    "source_checkpoint_sha256",
    "dataset_split_hash",
    "extraction_code_sha256",
    "z_dim",
    "n_rows",
)


def is_stale(existing_manifest: dict | None, current_provenance: dict) -> bool:
    if existing_manifest is None:
        return True
    return any(existing_manifest.get(k) != current_provenance.get(k) for k in _STALENESS_KEYS)
