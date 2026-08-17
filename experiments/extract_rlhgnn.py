#!/usr/bin/env python
"""Phase 4: extract z_t + predictions for every Helpdesk test-set prefix of
A3 RLHGNN - see paper/phase4_representation_extraction.md for the full
schema/design.

Usage:
    uv run --extra torch-dgl python experiments/extract_rlhgnn.py [dataset]

Does not retrain: reloads from this model's own
results/<dataset>/rlhgnn/manifest.json (no-retrain pattern from
scripts/eval_sutran_val_dl.py). Skips recomputation if a cached
embeddings_manifest.json already reflects the current checkpoint/split/code.

z_t extraction point: model.encode(hg) - the graph-level max-pooled node
embedding (src/models/rlhgnn/model.py), architecturally distinct from every
other roster model (whole-prefix-graph pooling, not a last-position
readout). Unlike the padded-tensor models, RLHGNN's own adapter.encode_graphs
already covers the full test set directly (one DGL heterograph per prefix,
no fixed window) - batched here via dgl.batch in chunks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.prefixes import make_next_activity_prefixes  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from models.rlhgnn.adapter import build_vocab, encode_graphs  # noqa: E402
from models.rlhgnn.model import RLHGNNActivityOnly  # noqa: E402
from representations.cache import build_provenance, is_stale  # noqa: E402
from representations.common_fields import compute_common_fields  # noqa: E402

MODEL_NAME = "rlhgnn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return {pkg: _safe_version(pkg) for pkg in ("torch", "dgl", "numpy", "pandas")}


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
    df = load_dataset(REPO_ROOT / dataset_config["raw_path"], dataset_config["source"]["format"])
    split_cfg = dataset_config["split"]
    split = compute_split(df, split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"])
    parts = apply_split(df, split)
    assert split.split_hash == manifest["dataset_split_hash"], "split hash mismatch vs. training run"

    train_prefixes = make_next_activity_prefixes(parts["train"])
    test_prefixes = make_next_activity_prefixes(parts["test"])
    vocab = build_vocab(train_prefixes)
    assert len(test_prefixes) == manifest["n_test_prefixes"], "test prefix count mismatch vs. training run"

    provenance = build_provenance(
        dataset=args.dataset,
        model=MODEL_NAME,
        checkpoint_path=checkpoint_path,
        dataset_split_hash=split.split_hash,
        extraction_code_paths=SHARED_CODE_PATHS,
        z_dim=model_config["hidden_dim"],
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

    model = RLHGNNActivityOnly(
        vocab_size=vocab.vocab_size,
        num_classes=vocab.num_classes,
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
        num_layers=model_config["num_layers"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print(f"[{args.dataset}/{MODEL_NAME}] computing common (model-agnostic) fields...")
    common = compute_common_fields(parts["test"], test_prefixes)

    print(f"[{args.dataset}/{MODEL_NAME}] running z_t extraction over {len(test_prefixes)} test prefixes...")
    graphs, _labels = encode_graphs(test_prefixes, vocab)
    rev_class = {idx: name for name, idx in vocab.class_dict.items()}

    z_list, probs_list, conf_list, entropy_list, pred_list = [], [], [], [], []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            batch_graph = dgl.batch(graphs[i : i + batch_size]).to(DEVICE)
            z_t = model.encode(batch_graph)
            logits = model.classifier(z_t)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            top_prob, top_idx = probs.max(dim=-1)

            z_list.append(z_t.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            conf_list.append(top_prob.cpu().numpy())
            entropy_list.append(entropy.cpu().numpy())
            pred_list.extend(rev_class.get(int(c), "[UNK]") for c in top_idx.cpu().numpy())

    z = np.concatenate(z_list, axis=0)
    probs_arr = np.concatenate(probs_list, axis=0)
    conf = np.concatenate(conf_list, axis=0)
    entropy_arr = np.concatenate(entropy_list, axis=0)

    out = common.copy()
    out["dataset"] = args.dataset
    out["model"] = MODEL_NAME
    out["family"] = "A"
    out["objective"] = "next_event"
    out["z"] = z.tolist()
    out["event_probs"] = probs_arr.tolist()
    out["predicted_event"] = pred_list
    out["pred_confidence"] = conf
    out["entropy"] = entropy_arr
    out["correct"] = out["predicted_event"] == out["true_event"]
    out["predicted_suffix"] = None
    out["true_suffix"] = None
    out["suffix_dl_similarity"] = None

    out.to_parquet(embeddings_path, index=False)
    embeddings_manifest_path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(f"[{args.dataset}/{MODEL_NAME}] wrote {embeddings_path} ({len(out)} rows, z_dim={z.shape[1]})")


if __name__ == "__main__":
    main()
