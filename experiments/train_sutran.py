#!/usr/bin/env python
"""Train & evaluate A4 SuTraN (full-suffix, activity-only scope) on one
dataset.

Usage:
    uv run --extra torch python experiments/train_sutran.py configs/experiments/sutran_helpdesk.yaml

Deliberate deviations from the original repo (see
src/models/sutran/model.py's and adapter.py's docstrings for the full
list):
  1. Vocabulary built from TRAIN split only (same rationale as A1/A2).
  2. This project's own 64/16/20 time-based split is used, not SuTraN's
     own more elaborate overlap-aware out-of-time split procedure.
  3. Checkpoint selection uses teacher-forced VALIDATION LOSS per epoch
     (cheap), not full autoregressive generation + Damerau-Levenshtein
     similarity on the validation set every epoch (the original repo's own
     approach, which is more informative but far more expensive to run
     every epoch). Full autoregressive generation is still used for the
     final TEST set evaluation, where it matters most.
  4. Activity-only: no timestamp suffix or remaining-runtime prediction
     heads (see model.py's docstring for why this is a stricter
     simplification than the repo's own "NDA" variant, not an arbitrary
     cut-down).
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
from models.sutran.adapter import (  # noqa: E402
    build_reverse_class_dict,
    build_vocab,
    encode_prefixes,
    encode_training_suffixes,
    get_window_size,
    sos_tokens,
)
from models.sutran.model import SuTraNActivityOnly  # noqa: E402

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


def evaluate_suffix_generation(model, prefix_df, vocab, window_size, batch_size=256) -> pd.DataFrame:
    """Autoregressive generation + normalized DL similarity per instance,
    grouped by prefix length k."""
    model.eval()
    rev = build_reverse_class_dict(vocab)
    all_sims, all_k = [], []

    x_all, pad_all = encode_prefixes(prefix_df, vocab, window_size)
    sos_all = sos_tokens(prefix_df, vocab)
    ks = prefix_df["k"].to_numpy()
    suffixes = prefix_df["suffix"].tolist()

    with torch.no_grad():
        for i in range(0, len(prefix_df), batch_size):
            x = x_all[i : i + batch_size].to(DEVICE)
            pad = pad_all[i : i + batch_size].to(DEVICE)
            sos = sos_all[i : i + batch_size].to(DEVICE)
            pred_classes = model.generate(x, pad, sos, max_len=window_size, eos_class=vocab.eos_class)
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
    df = load_dataset(REPO_ROOT / dataset_config["raw_path"], dataset_config["source"]["format"], dataset_config.get("date_filter"))
    split_cfg = dataset_config["split"]
    split = compute_split(df, split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"])
    parts = apply_split(df, split)

    print(f"[{dataset_name}] building suffix prefixes + vocab from TRAIN split only...")
    train_prefixes = make_suffix_prefixes(parts["train"])
    val_prefixes = make_suffix_prefixes(parts["val"])
    test_prefixes = make_suffix_prefixes(parts["test"])
    vocab = build_vocab(train_prefixes)
    window_size = get_window_size(train_prefixes)
    print(
        f"[{dataset_name}] vocab_size={vocab.vocab_size} num_classes={vocab.num_classes} "
        f"window_size={window_size} "
        f"train={len(train_prefixes)} val={len(val_prefixes)} test={len(test_prefixes)}"
    )

    train_x, train_pad = encode_prefixes(train_prefixes, vocab, window_size)
    train_dec_in, train_y = encode_training_suffixes(train_prefixes, vocab, window_size)
    val_x, val_pad = encode_prefixes(val_prefixes, vocab, window_size)
    val_dec_in, val_y = encode_training_suffixes(val_prefixes, vocab, window_size)

    model = SuTraNActivityOnly(
        vocab_size=vocab.vocab_size,
        num_classes=vocab.num_classes,
        d_model=model_config["d_model"],
        num_prefix_encoder_layers=model_config["num_prefix_encoder_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        dropout=model_config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config["learning_rate"], weight_decay=model_config["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=model_config["lr_decay_gamma"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    train_loader = DataLoader(
        TensorDataset(train_x, train_pad, train_dec_in, train_y),
        batch_size=model_config["batch_size"],
        shuffle=True,
    )

    result_dir = REPO_ROOT / "results" / dataset_name / "sutran"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "checkpoint.pt"
    resume_state_path = result_dir / "resume_state.pt"

    total_epochs = model_config["epochs"]
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    stopped_early = False

    # Checkpoint/resume, same pattern as train_mlmme.py: full training state
    # (model, optimizer, LR scheduler, epoch counter, patience counter, RNG
    # states) snapshotted to resume_state.pt after every epoch, so
    # re-running this exact command resumes with bit-for-bit RNG
    # continuity instead of restarting - needed once BPIC17/BPIC19-scale
    # training runs long enough to need interrupting (found necessary for
    # A7 MLMME on Helpdesk already; SuTraN had no such support until a user
    # run on Sepsis raised the question).
    if resume_state_path.exists():
        print(f"[{dataset_name}] found {resume_state_path}, resuming training state...")
        resume_state = torch.load(resume_state_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        lr_scheduler.load_state_dict(resume_state["scheduler_state"])
        start_epoch = resume_state["next_epoch"]
        best_val_loss = resume_state["best_val_loss"]
        epochs_without_improvement = resume_state["epochs_without_improvement"]
        train_losses = resume_state["train_losses"]
        stopped_early = resume_state["stopped_early"]
        random.setstate(resume_state["random_state"])
        np.random.set_state(resume_state["numpy_state"])
        torch.set_rng_state(resume_state["torch_state"].cpu())
        print(
            f"[{dataset_name}] resumed at epoch {start_epoch} "
            f"(best_val_loss={best_val_loss:.4f}, {len(train_losses)} epochs already run, "
            f"stopped_early={stopped_early})"
        )

    def save_resume_state(next_epoch: int, stopped_early_flag: bool) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
                "next_epoch": next_epoch,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "train_losses": train_losses,
                "stopped_early": stopped_early_flag,
                "random_state": random.getstate(),
                "numpy_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
            },
            resume_state_path,
        )

    if stopped_early:
        print(f"[{dataset_name}] training already early-stopped in a prior run, skipping straight to evaluation.")
    elif start_epoch >= total_epochs:
        print(f"[{dataset_name}] already ran the requested {total_epochs} epochs, skipping straight to evaluation.")
    else:
        print(
            f"[{dataset_name}] training epochs {start_epoch}..{total_epochs - 1} "
            f"(patience={model_config['patience']})..."
        )

    for epoch in (range(start_epoch, total_epochs) if not stopped_early else []):
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
        lr_scheduler.step()
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        val_loss = teacher_forced_val_loss(model, val_x, val_pad, val_dec_in, val_y, loss_fn)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= model_config["patience"]:
            print(f"No val_loss improvement for {model_config['patience']} epochs. Stopping at epoch {epoch}.")
            save_resume_state(next_epoch=epoch + 1, stopped_early_flag=True)
            break
        else:
            save_resume_state(next_epoch=epoch + 1, stopped_early_flag=False)

    print(f"[{dataset_name}] restoring best (val-loss) checkpoint for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    print(f"[{dataset_name}] running autoregressive suffix generation on the test set...")
    per_k, overall_dl_similarity = evaluate_suffix_generation(model, test_prefixes, vocab, window_size)
    per_k.to_csv(result_dir / "test_metrics_by_prefix_length.csv", index=False)
    print(f"[{dataset_name}] overall test mean normalized DL similarity: {overall_dl_similarity:.4f}")

    manifest = {
        "model": "sutran",
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
        "n_train_prefixes": len(train_prefixes),
        "n_val_prefixes": len(val_prefixes),
        "n_test_prefixes": len(test_prefixes),
        "epochs_run": len(train_losses),
        "final_train_loss": train_losses[-1],
        "best_val_loss": best_val_loss,
        "overall_test_mean_dl_similarity": float(overall_dl_similarity),
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[{dataset_name}] wrote {result_dir} (checkpoint, per-k metrics, manifest.json)")


if __name__ == "__main__":
    main()
