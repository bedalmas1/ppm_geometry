#!/usr/bin/env python
"""Train & evaluate B2 controlled Transformer (Family B, full-suffix
objective) on one dataset.

Usage:
    uv run --extra torch python experiments/train_controlled_transformer_suffix.py configs/experiments/controlled_transformer_suffix_helpdesk.yaml

Second of the two Family B controlled variants: reuses
`ControlledTransformerEncoder` UNCHANGED from B1
(models.controlled_transformer.model), only the decoder/head differs.
configs/models/controlled_transformer_suffix.yaml mirrors B1's
architecture/training-budget hyperparameters verbatim - see that file's
comment - so the B1-vs-B2 comparison isolates objective from architecture,
per spec §5. Trained on the same shared suffix-prefix definition
(data.prefixes.make_suffix_prefixes) as A4/A5, on this project's own split.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.prefixes import EOS, make_suffix_prefixes  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from evaluation.suffix_metrics import normalized_dl_similarity  # noqa: E402
from models.controlled_transformer.model import ControlledTransformerSuffix  # noqa: E402
from models.controlled_transformer.suffix_adapter import (  # noqa: E402
    build_reverse_class_dict,
    build_vocab,
    encode_prefixes,
    encode_training_suffixes,
    get_prefix_window,
    get_suffix_window,
    sos_tokens,
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
    return {pkg: _safe_version(pkg) for pkg in ("torch", "numpy", "pandas")}


def teacher_forced_val_loss(model, val_x, val_pad, val_dec_in, val_y, loss_fn, batch_size=256) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_x), batch_size):
            x = val_x[i : i + batch_size].to(DEVICE)
            pad = val_pad[i : i + batch_size].to(DEVICE)
            dec_in = val_dec_in[i : i + batch_size].to(DEVICE)
            y = val_y[i : i + batch_size].to(DEVICE)
            logits = model(x, pad, dec_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / n


def evaluate_suffix_generation(model, prefix_df, vocab, prefix_window, suffix_window, batch_size=256) -> pd.DataFrame:
    """Autoregressive generation + normalized DL similarity per instance,
    grouped by prefix length k."""
    model.eval()
    rev = build_reverse_class_dict(vocab)
    all_sims, all_k = [], []

    x_all, pad_all = encode_prefixes(prefix_df, vocab, prefix_window)
    sos_all = sos_tokens(prefix_df, vocab)
    ks = prefix_df["k"].to_numpy()
    suffixes = prefix_df["suffix"].tolist()

    with torch.no_grad():
        for i in range(0, len(prefix_df), batch_size):
            x = x_all[i : i + batch_size].to(DEVICE)
            pad = pad_all[i : i + batch_size].to(DEVICE)
            sos = sos_all[i : i + batch_size].to(DEVICE)
            pred_classes = model.generate(x, pad, sos, max_len=suffix_window, eos_class=vocab.eos_class)
            pred_classes = pred_classes.cpu().numpy()

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

    print(f"[{dataset_name}] building suffix prefixes + vocab from TRAIN split only...")
    train_prefixes = make_suffix_prefixes(parts["train"])
    val_prefixes = make_suffix_prefixes(parts["val"])
    test_prefixes = make_suffix_prefixes(parts["test"])
    vocab = build_vocab(train_prefixes)
    prefix_window = get_prefix_window(train_prefixes)
    suffix_window = get_suffix_window(train_prefixes)
    print(
        f"[{dataset_name}] vocab_size={vocab.vocab_size} num_classes={vocab.num_classes} "
        f"prefix_window={prefix_window} suffix_window={suffix_window} "
        f"train={len(train_prefixes)} val={len(val_prefixes)} test={len(test_prefixes)}"
    )

    train_x, train_pad = encode_prefixes(train_prefixes, vocab, prefix_window)
    train_dec_in, train_y = encode_training_suffixes(train_prefixes, vocab, suffix_window)
    val_x, val_pad = encode_prefixes(val_prefixes, vocab, prefix_window)
    val_dec_in, val_y = encode_training_suffixes(val_prefixes, vocab, suffix_window)

    model = ControlledTransformerSuffix(
        vocab_size=vocab.vocab_size,
        num_classes=vocab.num_classes,
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        d_ff=model_config["d_ff"],
        dropout=model_config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config["learning_rate"], weight_decay=model_config["weight_decay"]
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    train_loader = DataLoader(
        TensorDataset(train_x, train_pad, train_dec_in, train_y),
        batch_size=model_config["batch_size"],
        shuffle=True,
    )

    result_dir = REPO_ROOT / "results" / dataset_name / "controlled_transformer_suffix"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "checkpoint.pt"

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []

    print(f"[{dataset_name}] training for up to {model_config['epochs']} epochs (patience={model_config['patience']})...")
    for epoch in range(model_config["epochs"]):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for x, pad, dec_in, y in train_loader:
            x, pad, dec_in, y = x.to(DEVICE), pad.to(DEVICE), dec_in.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x, pad, dec_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_config["max_grad_norm"])
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        val_loss = teacher_forced_val_loss(model, val_x, val_pad, val_dec_in, val_y, loss_fn)
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

    print(f"[{dataset_name}] running autoregressive suffix generation on the validation set...")
    val_per_k, overall_val_dl_similarity = evaluate_suffix_generation(model, val_prefixes, vocab, prefix_window, suffix_window)
    print(f"[{dataset_name}] overall val mean normalized DL similarity: {overall_val_dl_similarity:.4f}")

    print(f"[{dataset_name}] running autoregressive suffix generation on the test set...")
    per_k, overall_dl_similarity = evaluate_suffix_generation(model, test_prefixes, vocab, prefix_window, suffix_window)
    per_k.to_csv(result_dir / "test_metrics_by_prefix_length.csv", index=False)
    print(f"[{dataset_name}] overall test mean normalized DL similarity: {overall_dl_similarity:.4f}")

    manifest = {
        "model": "controlled_transformer_suffix",
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
        "prefix_window": prefix_window,
        "suffix_window": suffix_window,
        "n_train_prefixes": len(train_prefixes),
        "n_val_prefixes": len(val_prefixes),
        "n_test_prefixes": len(test_prefixes),
        "epochs_run": len(train_losses),
        "final_train_loss": train_losses[-1],
        "best_val_loss": best_val_loss,
        "overall_val_mean_dl_similarity": float(overall_val_dl_similarity),
        "overall_test_mean_dl_similarity": float(overall_dl_similarity),
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[{dataset_name}] wrote {result_dir} (checkpoint, per-k metrics, manifest.json)")


if __name__ == "__main__":
    main()
