#!/usr/bin/env python
"""Train & evaluate B1 controlled Transformer (Family B, next-event
objective) on one dataset.

Usage:
    uv run --extra torch python experiments/train_controlled_transformer_next.py configs/experiments/controlled_transformer_next_helpdesk.yaml

Unlike every Family A model, there is no external repo to adapt here: this
is this project's own from-scratch, controlled Transformer encoder (spec
§5), trained on the same shared next-event prefix definition
(data.prefixes.make_next_activity_prefixes) as A1/A2, on this project's own
64/16/20 time-based split. Its architecture/training-budget hyperparameters
(configs/models/controlled_transformer_next.yaml) are the ones B2
(full-suffix, to be built next) must copy verbatim for a genuinely
controlled, objective-only comparison.
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
from data.splits import apply_split, compute_split  # noqa: E402
from models.controlled_transformer.adapter import build_vocab, encode, get_max_case_length, make_prefixes  # noqa: E402
from models.controlled_transformer.model import ControlledTransformerNextEvent  # noqa: E402

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


def val_loss_and_accuracy(model, val_x, val_pad, val_len, val_y, loss_fn, batch_size=256) -> tuple[float, float]:
    model.eval()
    total_loss, n_correct, n = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(val_x), batch_size):
            x = val_x[i : i + batch_size].to(DEVICE)
            pad = val_pad[i : i + batch_size].to(DEVICE)
            length = val_len[i : i + batch_size].to(DEVICE)
            y = val_y[i : i + batch_size].to(DEVICE)
            logits = model(x, pad, length)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            n_correct += (logits.argmax(dim=-1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, n_correct / n


def evaluate_by_prefix_length(model, prefix_df: pd.DataFrame, vocab, max_case_length: int) -> pd.DataFrame:
    from sklearn import metrics as skmetrics

    rows = []
    for k, group in prefix_df.groupby("k"):
        x, pad, length, y = encode(group, vocab, max_case_length)
        model.eval()
        with torch.no_grad():
            logits = model(x.to(DEVICE), pad.to(DEVICE), length.to(DEVICE))
        y_pred = logits.argmax(dim=-1).cpu().numpy()
        y_true = y.numpy()
        rows.append(
            {
                "k": int(k),
                "n": len(group),
                "accuracy": skmetrics.accuracy_score(y_true, y_pred),
                "f1_weighted": skmetrics.f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "precision_weighted": skmetrics.precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall_weighted": skmetrics.recall_score(y_true, y_pred, average="weighted", zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


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

    print(f"[{dataset_name}] building next-event prefixes + vocab from TRAIN split only...")
    train_prefixes = make_prefixes(parts["train"])
    val_prefixes = make_prefixes(parts["val"])
    test_prefixes = make_prefixes(parts["test"])
    vocab = build_vocab(train_prefixes)
    max_case_length = get_max_case_length(train_prefixes)
    print(
        f"[{dataset_name}] vocab_size={vocab.vocab_size} num_classes={vocab.num_classes} "
        f"max_case_length={max_case_length} train_prefixes={len(train_prefixes)} "
        f"val_prefixes={len(val_prefixes)} test_prefixes={len(test_prefixes)}"
    )

    train_x, train_pad, train_len, train_y = encode(train_prefixes, vocab, max_case_length)
    val_x, val_pad, val_len, val_y = encode(val_prefixes, vocab, max_case_length)

    model = ControlledTransformerNextEvent(
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
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(train_x, train_pad, train_len, train_y),
        batch_size=model_config["batch_size"],
        shuffle=True,
    )

    result_dir = REPO_ROOT / "results" / dataset_name / "controlled_transformer_next"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "checkpoint.pt"
    resume_state_path = result_dir / "resume_state.pt"

    total_epochs = model_config["epochs"]
    start_epoch = 0
    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    train_losses = []
    stopped_early = False

    # Checkpoint/resume, same pattern as train_mlmme.py/train_sutran.py:
    # full training state snapshotted to resume_state.pt after every epoch,
    # so re-running this exact command resumes with bit-for-bit RNG
    # continuity instead of restarting.
    if resume_state_path.exists():
        print(f"[{dataset_name}] found {resume_state_path}, resuming training state...")
        resume_state = torch.load(resume_state_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = resume_state["next_epoch"]
        best_val_accuracy = resume_state["best_val_accuracy"]
        epochs_without_improvement = resume_state["epochs_without_improvement"]
        train_losses = resume_state["train_losses"]
        stopped_early = resume_state["stopped_early"]
        random.setstate(resume_state["random_state"])
        np.random.set_state(resume_state["numpy_state"])
        torch.set_rng_state(resume_state["torch_state"].cpu())
        print(
            f"[{dataset_name}] resumed at epoch {start_epoch} "
            f"(best_val_accuracy={best_val_accuracy:.4f}, {len(train_losses)} epochs already run, "
            f"stopped_early={stopped_early})"
        )

    def save_resume_state(next_epoch: int, stopped_early_flag: bool) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_epoch": next_epoch,
                "best_val_accuracy": best_val_accuracy,
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
        for x, pad, length, y in train_loader:
            x, pad, length, y = x.to(DEVICE), pad.to(DEVICE), length.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x, pad, length)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_config["max_grad_norm"])
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        val_loss, val_accuracy = val_loss_and_accuracy(model, val_x, val_pad, val_len, val_y, loss_fn)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}", flush=True)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= model_config["patience"]:
            print(f"No val_accuracy improvement for {model_config['patience']} epochs. Stopping at epoch {epoch}.")
            save_resume_state(next_epoch=epoch + 1, stopped_early_flag=True)
            break
        else:
            save_resume_state(next_epoch=epoch + 1, stopped_early_flag=False)

    print(f"[{dataset_name}] restoring best (val-accuracy) checkpoint for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    per_k = evaluate_by_prefix_length(model, test_prefixes, vocab, max_case_length)
    per_k.to_csv(result_dir / "test_metrics_by_prefix_length.csv", index=False)

    test_x, test_pad, test_len, test_y = encode(test_prefixes, vocab, max_case_length)
    model.eval()
    with torch.no_grad():
        y_pred_all = model(test_x.to(DEVICE), test_pad.to(DEVICE), test_len.to(DEVICE)).argmax(dim=-1).cpu().numpy()
    from sklearn import metrics as skmetrics

    overall = {
        "macro_avg_accuracy_across_k": float(per_k["accuracy"].mean()),
        "macro_avg_f1_weighted_across_k": float(per_k["f1_weighted"].mean()),
        "micro_accuracy": float(skmetrics.accuracy_score(test_y.numpy(), y_pred_all)),
        "micro_f1_weighted": float(skmetrics.f1_score(test_y.numpy(), y_pred_all, average="weighted", zero_division=0)),
    }
    print(f"[{dataset_name}] overall test metrics: {overall}")

    manifest = {
        "model": "controlled_transformer_next",
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
        "max_case_length": max_case_length,
        "n_train_prefixes": len(train_prefixes),
        "n_val_prefixes": len(val_prefixes),
        "n_test_prefixes": len(test_prefixes),
        "epochs_run": len(train_losses),
        "final_train_loss": train_losses[-1],
        "best_val_accuracy": best_val_accuracy,
        "overall_test_metrics": overall,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[{dataset_name}] wrote {result_dir} (checkpoint, per-k metrics, manifest.json)")


if __name__ == "__main__":
    main()
