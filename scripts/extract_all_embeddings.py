#!/usr/bin/env python
"""Run all 9 roster models' Phase 4 extraction scripts in sequence, on one
dataset. Convenience wrapper, same spirit as scripts/prepare_dataset.py's
--all flag - each underlying experiments/extract_<model>.py is independently
runnable and already skips recomputation if its cache is fresh (see
src/representations/cache.py), so re-running this script is cheap.

Usage:
    python scripts/extract_all_embeddings.py [dataset]

Each model needs its own optional-dependency group installed (tf / torch /
torch-hf / torch-dgl - see pyproject.toml), so this script shells out to
`uv run --extra <group> python experiments/extract_<model>.py <dataset>`
rather than importing everything into one process (TensorFlow and the
`torch-dgl` group's pinned torch build are not safe to import alongside the
plain `torch` group in the same interpreter - see STATUS.md's Phase 2 notes
on the `[tool.uv] conflicts` extras).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (script name, uv extra) - order mirrors Phase 3's own integration order.
MODELS = [
    ("extract_process_transformer.py", "tf"),
    ("extract_generative_lstm.py", "tf"),
    ("extract_rlhgnn.py", "torch-dgl"),
    ("extract_sutran.py", "torch"),
    ("extract_crtp_lstm.py", "torch"),
    ("extract_lupin.py", "torch-hf"),
    ("extract_mlmme.py", "torch"),
    ("extract_controlled_transformer_next.py", "torch"),
    ("extract_controlled_transformer_suffix.py", "torch"),
]


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "helpdesk"
    failures = []
    for script_name, extra in MODELS:
        script_path = REPO_ROOT / "experiments" / script_name
        print(f"\n=== {script_name} ({dataset}) ===")
        result = subprocess.run(
            ["uv", "run", "--extra", extra, "python", str(script_path), dataset],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            failures.append(script_name)
            print(f"!!! {script_name} failed (exit {result.returncode}), continuing with the rest.")

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s): {failures}")
        sys.exit(1)
    print("\nAll 9 models' embeddings extracted (or already up to date).")


if __name__ == "__main__":
    main()
