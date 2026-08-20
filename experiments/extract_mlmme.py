#!/usr/bin/env python
"""Phase 4: extract z_t + predictions for every Helpdesk test-set prefix of
A7 MLMME - see paper/phase4_representation_extraction.md for the full
schema/design.

**LICENSE NOTICE**: imports from src/models/mlmme/, derived from
https://github.com/farbodtaymouri/MLMME (GPL-3.0) - see that module's
docstring for the full isolation notice.

Usage:
    uv run --extra torch python experiments/extract_mlmme.py [dataset]

Does not retrain: reloads from this model's own
results/<dataset>/mlmme/manifest.json (no-retrain pattern from
scripts/eval_sutran_val_dl.py) - only `checkpoint.pt` (the generator's
best-val-loss state_dict) is needed, not `resume_state.pt`'s full GAN
training state. Skips recomputation if a cached embeddings_manifest.json
already reflects the current checkpoint/split/code.

z_t extraction point: model.encode(prefix_onehot, prefix_lengths) - the
encoder's final top-layer hidden state (src/models/mlmme/model.py).
"Prediction probabilities"/entropy use the first decoder step's output
(reusing the same encoder hidden/cell state computed for z_t, then one
Decoder.forward call with the model's own literal-1.0s start-of-sequence
input - exactly generate()'s own first iteration), softmax'd - the decoder's
raw output is ReLU'd, not softmax'd, per the documented paper/code
discrepancy in model.py, and this project's own CrossEntropyLoss during
training already treats those ReLU'd values as softmax logits internally, so
this is applying the same convention, not inventing a new one.
predicted_suffix/true_suffix/suffix_dl_similarity reuse model.generate() and
evaluation.suffix_metrics.normalized_dl_similarity.
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
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.prefixes import EOS, make_suffix_prefixes  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from evaluation.suffix_metrics import normalized_dl_similarity  # noqa: E402
from models.mlmme.adapter import encode_prefixes_classidx  # noqa: E402
from models.mlmme.model import MLMMEGenerator  # noqa: E402
from models.sutran.adapter import build_reverse_class_dict, build_vocab, get_window_size  # noqa: E402
from representations.cache import build_provenance, is_stale  # noqa: E402
from representations.common_fields import compute_common_fields  # noqa: E402

MODEL_NAME = "mlmme"
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
    return {pkg: _safe_version(pkg) for pkg in ("torch", "numpy", "pandas")}


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

    train_prefixes = make_suffix_prefixes(parts["train"])
    test_prefixes = make_suffix_prefixes(parts["test"])
    vocab = build_vocab(train_prefixes)
    window_size = get_window_size(train_prefixes)
    assert window_size == manifest["window_size"], "window_size mismatch vs. training run"
    assert len(test_prefixes) == manifest["n_test_prefixes"], "test prefix count mismatch vs. training run"

    provenance = build_provenance(
        dataset=args.dataset,
        model=MODEL_NAME,
        checkpoint_path=checkpoint_path,
        dataset_split_hash=split.split_hash,
        extraction_code_paths=SHARED_CODE_PATHS,
        z_dim=model_config["hidden_size"],
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

    model = MLMMEGenerator(
        num_classes=vocab.num_classes,
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print(f"[{args.dataset}/{MODEL_NAME}] computing common (model-agnostic) fields...")
    common = compute_common_fields(parts["test"], test_prefixes)

    print(f"[{args.dataset}/{MODEL_NAME}] running z_t extraction over {len(test_prefixes)} test prefixes...")
    prefix_idx, lengths = encode_prefixes_classidx(test_prefixes, vocab, window_size)
    rev_class = build_reverse_class_dict(vocab)
    true_suffixes = [[a for a in suf if a != EOS] for suf in test_prefixes["suffix"]]

    z_list, probs_list, conf_list, entropy_list, pred_list = [], [], [], [], []
    predicted_suffixes, suffix_sims = [], []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(test_prefixes), batch_size):
            idx = prefix_idx[i : i + batch_size].to(DEVICE)
            lens = lengths[i : i + batch_size].to(DEVICE)
            prefix_onehot = F.one_hot(idx, vocab.num_classes).float()
            batch_len = idx.size(0)

            h, c = model.encoder(prefix_onehot, lens)
            z_t = h[-1]
            sos_input = torch.ones(batch_len, 1, vocab.num_classes, device=DEVICE)
            first_out, _, _ = model.decoder(sos_input, h, c)
            first_probs = F.softmax(first_out.squeeze(1), dim=-1)
            entropy = -(first_probs * first_probs.clamp_min(1e-12).log()).sum(dim=-1)
            top_prob, top_idx = first_probs.max(dim=-1)

            pred_classes = model.generate(prefix_onehot, lens, max_len=window_size).cpu().numpy()

            z_list.append(z_t.cpu().numpy())
            probs_list.append(first_probs.cpu().numpy())
            conf_list.append(top_prob.cpu().numpy())
            entropy_list.append(entropy.cpu().numpy())
            pred_list.extend(rev_class.get(int(c_), "[UNK]") for c_ in top_idx.cpu().numpy())

            for row_idx in range(pred_classes.shape[0]):
                global_idx = i + row_idx
                pred_seq = []
                for c_ in pred_classes[row_idx]:
                    name = rev_class.get(int(c_), "[UNK]")
                    if name == EOS:
                        break
                    pred_seq.append(name)
                predicted_suffixes.append(pred_seq)
                suffix_sims.append(normalized_dl_similarity(pred_seq, true_suffixes[global_idx]))

    z = np.concatenate(z_list, axis=0)
    probs_arr = np.concatenate(probs_list, axis=0)
    conf = np.concatenate(conf_list, axis=0)
    entropy_arr = np.concatenate(entropy_list, axis=0)

    out = common.copy()
    out["dataset"] = args.dataset
    out["model"] = MODEL_NAME
    out["family"] = "A"
    out["objective"] = "full_suffix"
    out["z"] = z.tolist()
    out["event_probs"] = probs_arr.tolist()
    out["predicted_event"] = pred_list
    out["pred_confidence"] = conf
    out["entropy"] = entropy_arr
    out["correct"] = out["predicted_event"] == out["true_event"]
    out["predicted_suffix"] = predicted_suffixes
    out["true_suffix"] = true_suffixes
    out["suffix_dl_similarity"] = suffix_sims

    out.to_parquet(embeddings_path, index=False)
    embeddings_manifest_path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(f"[{args.dataset}/{MODEL_NAME}] wrote {embeddings_path} ({len(out)} rows, z_dim={z.shape[1]})")


if __name__ == "__main__":
    main()
