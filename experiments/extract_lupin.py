#!/usr/bin/env python
"""Phase 4: extract z_t + predictions for every Helpdesk test-set prefix of
A6 LUPIN - see paper/phase4_representation_extraction.md for the full
schema/design.

**LICENSE NOTICE**: imports from src/models/lupin/, derived from
https://github.com/vinspdb/LUPIN (CC BY-NC-SA 4.0) - see that module's
docstring for the full isolation notice.

Usage:
    uv run --extra torch-hf python experiments/extract_lupin.py [dataset]

Does not retrain: reloads from this model's own
results/<dataset>/lupin/manifest.json (no-retrain pattern from
scripts/eval_sutran_val_dl.py). Skips recomputation if a cached
embeddings_manifest.json already reflects the current checkpoint/split/code.

z_t extraction point: model.encode(input_ids, attention_mask) - BERT's
pooled [CLS] representation, already one vector per prefix (no
length-indexed readout needed - LUPIN encodes the whole prefix as one text
sequence). "Prediction probabilities"/entropy use the first output head's
softmax (the immediate next event); predicted_suffix uses every head's
argmax, truncated at the first predicted EOS, same convention as
train_lupin.py's own evaluate_direct.
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
from models.lupin.adapter import build_prefix_texts, encode_prefix_texts  # noqa: E402
from models.lupin.model import LupinActivityOnly  # noqa: E402
from models.sutran.adapter import build_reverse_class_dict, build_vocab, get_window_size  # noqa: E402
from representations.cache import build_provenance, is_stale  # noqa: E402
from representations.common_fields import compute_common_fields  # noqa: E402

MODEL_NAME = "lupin"
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
    return {pkg: _safe_version(pkg) for pkg in ("torch", "transformers", "numpy", "pandas")}


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
    max_token_length = manifest["max_token_length"]

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

    from transformers import BertConfig, BertTokenizerFast

    bert_hidden_size = BertConfig.from_pretrained(model_config["bert_model_name"]).hidden_size

    provenance = build_provenance(
        dataset=args.dataset,
        model=MODEL_NAME,
        checkpoint_path=checkpoint_path,
        dataset_split_hash=split.split_hash,
        extraction_code_paths=SHARED_CODE_PATHS,
        z_dim=bert_hidden_size,
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

    model = LupinActivityOnly(
        num_classes=vocab.num_classes,
        num_heads=window_size,
        bert_model_name=model_config["bert_model_name"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print(f"[{args.dataset}/{MODEL_NAME}] computing common (model-agnostic) fields...")
    common = compute_common_fields(parts["test"], test_prefixes)

    print(f"[{args.dataset}/{MODEL_NAME}] building activity-only text stories + tokenizing...")
    test_texts = build_prefix_texts(test_prefixes)
    tokenizer = BertTokenizerFast.from_pretrained(model_config["bert_model_name"], truncation_side="left")
    input_ids, attention_mask = encode_prefix_texts(test_texts, tokenizer, max_token_length)

    print(f"[{args.dataset}/{MODEL_NAME}] running z_t extraction over {len(test_prefixes)} test prefixes...")
    rev_class = build_reverse_class_dict(vocab)
    true_suffixes = [[a for a in suf if a != EOS] for suf in test_prefixes["suffix"]]

    z_list, probs_list, conf_list, entropy_list, pred_list = [], [], [], [], []
    predicted_suffixes, suffix_sims = [], []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(test_prefixes), batch_size):
            ii = input_ids[i : i + batch_size].to(DEVICE)
            am = attention_mask[i : i + batch_size].to(DEVICE)
            z_t = model.encode(ii, am)
            logits = torch.stack([head(z_t) for head in model.output_heads], dim=1)
            pred_classes = logits.argmax(dim=-1).cpu().numpy()  # (batch, num_heads)

            first_probs = F.softmax(logits[:, 0, :], dim=-1)
            entropy = -(first_probs * first_probs.clamp_min(1e-12).log()).sum(dim=-1)
            top_prob, top_idx = first_probs.max(dim=-1)

            z_list.append(z_t.cpu().numpy())
            probs_list.append(first_probs.cpu().numpy())
            conf_list.append(top_prob.cpu().numpy())
            entropy_list.append(entropy.cpu().numpy())
            pred_list.extend(rev_class.get(int(c), "[UNK]") for c in top_idx.cpu().numpy())

            for row_idx in range(pred_classes.shape[0]):
                global_idx = i + row_idx
                pred_seq = []
                for c in pred_classes[row_idx]:
                    name = rev_class.get(int(c), "[UNK]")
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
