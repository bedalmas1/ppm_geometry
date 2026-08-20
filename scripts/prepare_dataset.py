#!/usr/bin/env python
"""Load one dataset config, split it, and write processed parquet + provenance.

Usage:
    uv run python scripts/prepare_dataset.py configs/datasets/sepsis.yaml
    uv run python scripts/prepare_dataset.py --all   # every configs/datasets/*.yaml

Never hardcodes a dataset's path/fractions — everything comes from the
config, per this project's config-driven-experiments convention.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from data.stats import compute_stats  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def software_versions() -> dict:
    versions = {}
    for pkg in ("pandas", "numpy", "pm4py"):
        try:
            versions[pkg] = version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def prepare(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = config["name"]
    print(f"[{name}] loading {config['raw_path']} (format={config['source']['format']})")

    df = load_dataset(REPO_ROOT / config["raw_path"], config["source"]["format"], config.get("date_filter"))

    stats = compute_stats(df)
    print(f"[{name}] {stats.n_cases} cases, {stats.n_events} events, {stats.n_activities} activities")

    split_cfg = config["split"]
    split = compute_split(
        df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
    )
    parts = apply_split(df, split)

    processed_dir = REPO_ROOT / config["processed_path"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    for part_name, part_df in parts.items():
        part_df.to_parquet(processed_dir / f"{part_name}.parquet", index=False)

    manifest = {
        "dataset": name,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "software_versions": software_versions(),
        "config": config,
        "split_hash": split.split_hash,
        "split_sizes": {part: len(cases) for part, cases in {
            "train": split.train_cases, "val": split.val_cases, "test": split.test_cases
        }.items()},
        "stats": stats.to_dict(),
    }
    (processed_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"[{name}] wrote {processed_dir} (train/val/test parquet + manifest.json)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, help="Path to a configs/datasets/*.yaml file")
    parser.add_argument("--all", action="store_true", help="Prepare every configs/datasets/*.yaml")
    args = parser.parse_args()

    if args.all:
        configs = sorted((REPO_ROOT / "configs" / "datasets").glob("*.yaml"))
    elif args.config:
        configs = [args.config]
    else:
        parser.error("pass a config path or --all")
        return

    for config_path in configs:
        prepare(config_path)


if __name__ == "__main__":
    main()
