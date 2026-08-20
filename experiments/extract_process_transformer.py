#!/usr/bin/env python
"""Phase 4: extract z_t + predictions for every Helpdesk test-set prefix of
A1 ProcessTransformer - see paper/phase4_representation_extraction.md for
the full schema/design.

Usage:
    uv run --extra tf python experiments/extract_process_transformer.py [dataset]

Does not retrain: reloads from this model's own
results/<dataset>/process_transformer/manifest.json (no-retrain pattern from
scripts/eval_sutran_val_dl.py). Skips recomputation if a cached
embeddings_manifest.json already reflects the current checkpoint/split/code.

z_t extraction point: the "prefix_representation" GlobalAveragePooling1D
layer (src/models/process_transformer/model.py) isn't exposed as a separate
model output today - this script builds a new functional
tf.keras.Model(inputs=model.input, outputs=[<that layer>, model.output])
reusing the loaded weights, without touching model.py (per its own
docstring's intent). The final Dense head uses activation="linear", so
softmax is applied here to get an actual probability distribution.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from models.process_transformer.adapter import build_vocab, encode, get_max_case_length, make_prefixes  # noqa: E402
from models.process_transformer.model import get_next_activity_model  # noqa: E402
from representations.cache import build_provenance, is_stale  # noqa: E402
from representations.common_fields import compute_common_fields  # noqa: E402

MODEL_NAME = "process_transformer"
THIS_FILE = Path(__file__).resolve()
SHARED_CODE_PATHS = [
    THIS_FILE,
    REPO_ROOT / "src" / "representations" / "common_fields.py",
    REPO_ROOT / "src" / "representations" / "cache.py",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _safe_version(pkg: str) -> str:
    try:
        return version(pkg)
    except Exception:
        return "unknown"


def software_versions() -> dict:
    return {pkg: _safe_version(pkg) for pkg in ("tensorflow", "keras", "numpy", "pandas")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="helpdesk")
    args = parser.parse_args()

    result_dir = REPO_ROOT / "results" / args.dataset / MODEL_NAME
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = REPO_ROOT / manifest["checkpoint_path"]
    embeddings_path = result_dir / "embeddings_test.parquet"
    embeddings_manifest_path = result_dir / "embeddings_manifest.json"

    dataset_config = yaml.safe_load(
        (REPO_ROOT / manifest["experiment_config"]["dataset_config"]).read_text(encoding="utf-8")
    )
    model_config = manifest["model_config"]

    print(f"[{args.dataset}/{MODEL_NAME}] reloading raw log + re-deriving split...")
    df = load_dataset(REPO_ROOT / dataset_config["raw_path"], dataset_config["source"]["format"], dataset_config.get("date_filter"))
    split_cfg = dataset_config["split"]
    split = compute_split(df, split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"])
    parts = apply_split(df, split)
    assert split.split_hash == manifest["dataset_split_hash"], "split hash mismatch vs. training run"

    train_prefixes = make_prefixes(parts["train"])
    test_prefixes = make_prefixes(parts["test"])
    vocab = build_vocab(train_prefixes)
    max_case_length = get_max_case_length(train_prefixes)
    assert max_case_length == manifest["max_case_length"], "max_case_length mismatch vs. training run"
    assert len(test_prefixes) == manifest["n_test_prefixes"], "test prefix count mismatch vs. training run"

    provenance = build_provenance(
        dataset=args.dataset,
        model=MODEL_NAME,
        checkpoint_path=checkpoint_path,
        dataset_split_hash=split.split_hash,
        extraction_code_paths=SHARED_CODE_PATHS,
        z_dim=model_config["embed_dim"],
        n_rows=len(test_prefixes),
        extracted_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit(),
        software_versions=software_versions(),
        output_path=embeddings_path.relative_to(REPO_ROOT),
    )
    existing = (
        json.loads(embeddings_manifest_path.read_text(encoding="utf-8"))
        if embeddings_manifest_path.exists()
        else None
    )
    if not is_stale(existing, provenance):
        print(f"[{args.dataset}/{MODEL_NAME}] embeddings cache up to date, skipping.")
        return

    import tensorflow as tf

    model = get_next_activity_model(
        max_case_length=max_case_length,
        vocab_size=vocab.vocab_size,
        output_dim=vocab.num_classes,
        embed_dim=model_config["embed_dim"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
    )
    model.load_weights(str(checkpoint_path))
    extractor = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer("prefix_representation").output, model.output],
    )

    print(f"[{args.dataset}/{MODEL_NAME}] computing common (model-agnostic) fields...")
    common = compute_common_fields(parts["test"], test_prefixes)

    print(f"[{args.dataset}/{MODEL_NAME}] running z_t extraction over {len(test_prefixes)} test prefixes...")
    token_x, _token_y = encode(test_prefixes, vocab, max_case_length)
    z, logits = extractor.predict(token_x, verbose=0)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1)
    top_idx = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    rev_class = {idx: name for name, idx in vocab.y_word_dict.items()}
    predicted_event = [rev_class.get(int(c), "[UNK]") for c in top_idx]

    out = common.copy()
    out["dataset"] = args.dataset
    out["model"] = MODEL_NAME
    out["family"] = "A"
    out["objective"] = "next_event"
    out["z"] = z.tolist()
    out["event_probs"] = probs.tolist()
    out["predicted_event"] = predicted_event
    out["pred_confidence"] = conf
    out["entropy"] = entropy
    out["correct"] = out["predicted_event"] == out["true_event"]
    out["predicted_suffix"] = None
    out["true_suffix"] = None
    out["suffix_dl_similarity"] = None

    out.to_parquet(embeddings_path, index=False)
    embeddings_manifest_path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(f"[{args.dataset}/{MODEL_NAME}] wrote {embeddings_path} ({len(out)} rows, z_dim={z.shape[1]})")


if __name__ == "__main__":
    main()
