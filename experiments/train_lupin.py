#!/usr/bin/env python
"""Train & evaluate A6 LUPIN (full-suffix, activity-only, text-narrative +
fine-tuned BERT, direct multi-head classification) on one dataset.

Usage:
    uv run --extra torch-hf python experiments/train_lupin.py configs/experiments/lupin_helpdesk.yaml

**LICENSE NOTICE**: this entrypoint imports from `src/models/lupin/`, which
is derived from https://github.com/vinspdb/LUPIN (CC BY-NC-SA 4.0) - see
`src/models/lupin/model.py`'s docstring for the full isolation notice.

Deliberate deviations from the original repo (see
src/models/lupin/{model.py,adapter.py}'s docstrings for the full list):
  1. Vocabulary (activity CLASS labels, not the BERT subword input
     vocabulary) built from TRAIN split only (same rationale as A1/A2/A4/A5).
  2. This project's own 64/16/20 time-based split is used, not the original
     repo's own 66/34 chronological split.
  3. Activity-only text stories (no resource/time/trace attributes) - see
     model.py's docstring for why this is a larger proportional cut for
     LUPIN than for any other roster model.
  4. `max_token_length` computed empirically from the TRAIN split then
     CAPPED at 48 tokens (not the repo's own MAX_LEN=512, and not even this
     project's own uncapped TRAIN maximum of 133+margin) - a measured,
     compute-budget-driven deviation (one epoch at the repo's own
     batch_size=8/uncapped max length measured ~59 min on this project's
     CPU-only hardware). The cap covers ~92-93% of TRAIN prefixes without
     any truncation; only the longest tail loses its oldest events (dropped
     via `truncation_side='left'`). `batch_size` (32, not the repo's 8) and
     `epochs` (capped at 15, not the repo's 50) are the same kind of
     disclosed compute-budget deviation - see configs/models/lupin.yaml's
     comments and paper/phase3_baseline_reproduction.md's A6 section for
     full measured-timing justification.
  5. Concrete `BertModel`/`BertTokenizerFast` classes used instead of
     `Auto*` wrappers - a real old-checkpoint/new-transformers-version
     compatibility gap caught by this project's pre-training smoke test,
     not a design choice.
  6. Checkpoint selection uses VALIDATION LOSS per epoch (masked
     cross-entropy over all suffix positions), same convention as A4/A5 -
     the original repo's own `train_llm` has a checkpoint-selection bug
     (`best_model = model` keeps a live reference, not a snapshot, so it
     silently ends up saving the LAST epoch's weights regardless of which
     epoch actually had the lowest validation loss) which is not
     replicated here.
  7. Evaluation uses this project's shared `evaluation.suffix_metrics.
     normalized_dl_similarity` (over the true list of activity-name
     tokens), not the original's own char-level DL distance over
     space-joined numeric class-ID strings - for direct comparability with
     A4/A5's numbers, per this project's project-wide convention.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.prefixes import EOS, make_suffix_prefixes  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from evaluation.suffix_metrics import normalized_dl_similarity  # noqa: E402
from models.lupin.adapter import (  # noqa: E402
    build_prefix_texts,
    encode_prefix_texts,
    max_encoded_length,
)
from models.lupin.model import LupinActivityOnly  # noqa: E402
from models.sutran.adapter import (  # noqa: E402
    build_reverse_class_dict,
    build_vocab,
    encode_training_suffixes,
    get_window_size,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def masked_val_loss(model, input_ids, attention_mask, y, loss_fn, batch_size=32) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            ii = input_ids[i : i + batch_size].to(DEVICE)
            am = attention_mask[i : i + batch_size].to(DEVICE)
            yb = y[i : i + batch_size].to(DEVICE)
            logits = model(ii, am)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            total += loss.item() * ii.size(0)
            n += ii.size(0)
    return total / n


def evaluate_direct(model, prefix_df, vocab, input_ids, attention_mask, batch_size=32) -> tuple[pd.DataFrame, float]:
    """Direct (single forward pass) suffix prediction + normalized DL
    similarity per instance, grouped by prefix length k - same convention as
    A5 CRTP-LSTM's evaluate_direct."""
    model.eval()
    rev = build_reverse_class_dict(vocab)
    suffixes = prefix_df["suffix"].tolist()
    ks = prefix_df["k"].to_numpy()
    all_sims, all_k = [], []

    with torch.no_grad():
        for i in range(0, len(prefix_df), batch_size):
            ii = input_ids[i : i + batch_size].to(DEVICE)
            am = attention_mask[i : i + batch_size].to(DEVICE)
            logits = model(ii, am)
            pred_classes = logits.argmax(dim=-1).cpu().numpy()  # (batch, num_heads)

            for row_idx in range(pred_classes.shape[0]):
                global_idx = i + row_idx
                pred_seq = []
                for c in pred_classes[row_idx]:
                    name = rev.get(int(c), "[UNK]")
                    if name == EOS:
                        break
                    pred_seq.append(name)
                true_seq = [a for a in suffixes[global_idx] if a != EOS]
                sim = normalized_dl_similarity(pred_seq, true_seq)
                all_sims.append(sim)
                all_k.append(int(ks[global_idx]))

    df = pd.DataFrame({"k": all_k, "dl_similarity": all_sims})
    per_k = df.groupby("k")["dl_similarity"].agg(["mean", "count"]).reset_index()
    per_k.columns = ["k", "mean_dl_similarity", "n"]
    return per_k, df["dl_similarity"].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_config", type=Path)
    args = parser.parse_args()

    exp_config = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    dataset_config = yaml.safe_load((REPO_ROOT / exp_config["dataset_config"]).read_text(encoding="utf-8"))
    model_config = yaml.safe_load((REPO_ROOT / exp_config["model_config"]).read_text(encoding="utf-8"))
    seed = exp_config["seed"]
    set_seed(seed)

    dataset_name = dataset_config["name"]
    print(f"[{dataset_name}] loading raw log and re-deriving this project's split...")
    df = load_dataset(REPO_ROOT / dataset_config["raw_path"], dataset_config["source"]["format"])
    split_cfg = dataset_config["split"]
    split = compute_split(df, split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"])
    parts = apply_split(df, split)

    print(f"[{dataset_name}] building suffix prefixes + activity-class vocab from TRAIN split only...")
    train_prefixes = make_suffix_prefixes(parts["train"])
    val_prefixes = make_suffix_prefixes(parts["val"])
    test_prefixes = make_suffix_prefixes(parts["test"])
    vocab = build_vocab(train_prefixes)
    window_size = get_window_size(train_prefixes)
    print(
        f"[{dataset_name}] vocab_size={vocab.vocab_size} num_classes={vocab.num_classes} "
        f"window_size={window_size} (num LUPIN output heads) "
        f"train={len(train_prefixes)} val={len(val_prefixes)} test={len(test_prefixes)}"
    )

    print(f"[{dataset_name}] building activity-only text stories (LUPIN input side)...")
    train_texts = build_prefix_texts(train_prefixes)
    val_texts = build_prefix_texts(val_prefixes)
    test_texts = build_prefix_texts(test_prefixes)

    tokenizer = BertTokenizerFast.from_pretrained(model_config["bert_model_name"], truncation_side="left")
    natural_max_token_length = max_encoded_length(train_texts, tokenizer) + model_config["max_token_length_margin"]
    max_token_length = min(natural_max_token_length, model_config["max_token_length_cap"])
    print(
        f"[{dataset_name}] max_token_length={max_token_length} "
        f"(TRAIN split's natural max+margin would be {natural_max_token_length}; "
        f"capped per configs/models/lupin.yaml's documented compute-budget rationale)"
    )

    train_input_ids, train_attention_mask = encode_prefix_texts(train_texts, tokenizer, max_token_length)
    val_input_ids, val_attention_mask = encode_prefix_texts(val_texts, tokenizer, max_token_length)
    test_input_ids, test_attention_mask = encode_prefix_texts(test_texts, tokenizer, max_token_length)

    _, train_y = encode_training_suffixes(train_prefixes, vocab, window_size)
    _, val_y = encode_training_suffixes(val_prefixes, vocab, window_size)

    model = LupinActivityOnly(
        num_classes=vocab.num_classes,
        num_heads=window_size,
        bert_model_name=model_config["bert_model_name"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    train_loader = DataLoader(
        TensorDataset(train_input_ids, train_attention_mask, train_y),
        batch_size=model_config["batch_size"],
        shuffle=True,
    )

    result_dir = REPO_ROOT / "results" / dataset_name / "lupin"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "checkpoint.pt"

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []

    print(f"[{dataset_name}] training for up to {model_config['epochs']} epochs (patience={model_config['patience']})...")
    for epoch in range(model_config["epochs"]):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for ii, am, y in train_loader:
            ii, am, y = ii.to(DEVICE), am.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ii, am)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        val_loss = masked_val_loss(model, val_input_ids, val_attention_mask, val_y, loss_fn)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= model_config["patience"]:
                print(f"No val_loss improvement for {model_config['patience']} epochs. Stopping at epoch {epoch}.")
                break

    print(f"[{dataset_name}] restoring best (val-loss) checkpoint for evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    print(f"[{dataset_name}] running direct suffix prediction on val + test sets...")
    val_per_k, overall_val_dl_similarity = evaluate_direct(model, val_prefixes, vocab, val_input_ids, val_attention_mask)
    test_per_k, overall_test_dl_similarity = evaluate_direct(model, test_prefixes, vocab, test_input_ids, test_attention_mask)
    val_per_k.to_csv(result_dir / "val_metrics_by_prefix_length.csv", index=False)
    test_per_k.to_csv(result_dir / "test_metrics_by_prefix_length.csv", index=False)
    print(f"[{dataset_name}] overall val mean normalized DL similarity: {overall_val_dl_similarity:.4f}")
    print(f"[{dataset_name}] overall test mean normalized DL similarity: {overall_test_dl_similarity:.4f}")

    manifest = {
        "model": "lupin",
        "dataset": dataset_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "software_versions": software_versions(),
        "device": str(DEVICE),
        "seed": seed,
        "experiment_config": exp_config,
        "model_config": model_config,
        "dataset_split_hash": split.split_hash,
        "vocab_size": vocab.vocab_size,
        "num_classes": vocab.num_classes,
        "window_size": window_size,
        "max_token_length": max_token_length,
        "n_train_prefixes": len(train_prefixes),
        "n_val_prefixes": len(val_prefixes),
        "n_test_prefixes": len(test_prefixes),
        "epochs_run": len(train_losses),
        "final_train_loss": train_losses[-1],
        "best_val_loss": best_val_loss,
        "overall_val_mean_dl_similarity": float(overall_val_dl_similarity),
        "overall_test_mean_dl_similarity": float(overall_test_dl_similarity),
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[{dataset_name}] wrote {result_dir} (checkpoint, per-k val/test metrics, manifest.json)")


if __name__ == "__main__":
    main()
