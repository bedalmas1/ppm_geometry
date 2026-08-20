#!/usr/bin/env python
"""Train & evaluate A3 RLHGNN (next-event, activity-only, fixed
"Comprehensive" heterogeneous graph - RL/DQN structure selection dropped)
on one dataset.

Usage:
    uv run --extra torch-dgl python experiments/train_rlhgnn.py configs/experiments/rlhgnn_helpdesk.yaml

Mirrors experiments/train_generative_lstm.py's overall structure
(train-only vocab, this project's own split, val-monitored checkpointing,
per-prefix-length test metrics) adapted for DGL heterogeneous graphs
instead of padded tensors - see src/models/rlhgnn/model.py and
src/models/rlhgnn/adapter.py for the model- and data-side scope decisions.
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

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from dgl.dataloading import GraphDataLoader
from sklearn import metrics as skmetrics

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.prefixes import make_next_activity_prefixes  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from models.rlhgnn.adapter import PrefixGraphDataset, build_vocab, encode_graphs  # noqa: E402
from models.rlhgnn.model import RLHGNNActivityOnly  # noqa: E402

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
    return {pkg: _safe_version(pkg) for pkg in ("torch", "dgl", "numpy", "pandas")}


def _run_eval(model, graphs, labels, batch_size=256) -> tuple[np.ndarray, np.ndarray, float]:
    """Batched forward pass over a (graphs, labels) pair; returns
    (y_true, y_pred, mean_loss)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            batch_graph = dgl.batch(graphs[i : i + batch_size]).to(DEVICE)
            batch_labels = labels[i : i + batch_size].to(DEVICE)
            logits = model(batch_graph)
            loss = loss_fn(logits, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            n += batch_labels.size(0)
            y_true.append(batch_labels.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), total_loss / n


def evaluate_by_prefix_length(model, prefix_df: pd.DataFrame, graphs, labels, batch_size=256) -> pd.DataFrame:
    model.eval()
    ks = prefix_df["k"].to_numpy()
    rows = []
    with torch.no_grad():
        for k in sorted(set(ks.tolist())):
            idx = np.where(ks == k)[0]
            group_graphs = [graphs[i] for i in idx]
            group_labels = labels[idx]
            y_true, y_pred, _ = _run_eval(model, group_graphs, group_labels, batch_size=batch_size)
            rows.append(
                {
                    "k": int(k),
                    "n": len(idx),
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

    print(f"[{dataset_name}] building next-activity prefixes + vocab from TRAIN split only...")
    train_prefixes = make_next_activity_prefixes(parts["train"])
    val_prefixes = make_next_activity_prefixes(parts["val"])
    test_prefixes = make_next_activity_prefixes(parts["test"])
    vocab = build_vocab(train_prefixes)
    print(
        f"[{dataset_name}] vocab_size={vocab.vocab_size} num_classes={vocab.num_classes} "
        f"train_prefixes={len(train_prefixes)} val_prefixes={len(val_prefixes)} test_prefixes={len(test_prefixes)}"
    )

    print(f"[{dataset_name}] building fixed 'Comprehensive' heterogeneous prefix graphs...")
    train_graphs, train_labels = encode_graphs(train_prefixes, vocab)
    val_graphs, val_labels = encode_graphs(val_prefixes, vocab)
    test_graphs, test_labels = encode_graphs(test_prefixes, vocab)

    model = RLHGNNActivityOnly(
        vocab_size=vocab.vocab_size,
        num_classes=vocab.num_classes,
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
        num_layers=model_config["num_layers"],
    ).to(DEVICE)

    optimizer = torch.optim.NAdam(model.parameters(), lr=model_config["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=model_config["lr_plateau_factor"], patience=model_config["lr_plateau_patience"]
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = GraphDataLoader(
        PrefixGraphDataset(train_graphs, train_labels), batch_size=model_config["batch_size"], shuffle=True
    )

    result_dir = REPO_ROOT / "results" / dataset_name / "rlhgnn"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "checkpoint.pt"
    resume_state_path = result_dir / "resume_state.pt"

    total_epochs = model_config["epochs"]
    start_epoch = 0
    best_val_acc = -1.0
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
        lr_scheduler.load_state_dict(resume_state["scheduler_state"])
        start_epoch = resume_state["next_epoch"]
        best_val_acc = resume_state["best_val_acc"]
        epochs_without_improvement = resume_state["epochs_without_improvement"]
        train_losses = resume_state["train_losses"]
        stopped_early = resume_state["stopped_early"]
        random.setstate(resume_state["random_state"])
        np.random.set_state(resume_state["numpy_state"])
        torch.set_rng_state(resume_state["torch_state"])
        print(
            f"[{dataset_name}] resumed at epoch {start_epoch} "
            f"(best_val_acc={best_val_acc:.4f}, {len(train_losses)} epochs already run, "
            f"stopped_early={stopped_early})"
        )

    def save_resume_state(next_epoch: int, stopped_early_flag: bool) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
                "next_epoch": next_epoch,
                "best_val_acc": best_val_acc,
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
        for batch_graph, batch_labels in train_loader:
            batch_graph = batch_graph.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_graph)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        y_true, y_pred, val_loss = _run_eval(model, val_graphs, val_labels)
        val_acc = skmetrics.accuracy_score(y_true, y_pred)
        lr_scheduler.step(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_acc:.4f}", flush=True)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
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

    print(f"[{dataset_name}] restoring best (val-accuracy) checkpoint for evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    per_k = evaluate_by_prefix_length(model, test_prefixes, test_graphs, test_labels)
    per_k.to_csv(result_dir / "test_metrics_by_prefix_length.csv", index=False)

    y_true_all, y_pred_all, _ = _run_eval(model, test_graphs, test_labels)
    overall = {
        "macro_avg_accuracy_across_k": float(per_k["accuracy"].mean()),
        "macro_avg_f1_weighted_across_k": float(per_k["f1_weighted"].mean()),
        "micro_accuracy": float(skmetrics.accuracy_score(y_true_all, y_pred_all)),
        "micro_f1_weighted": float(skmetrics.f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)),
    }
    print(f"[{dataset_name}] overall test metrics: {overall}")

    manifest = {
        "model": "rlhgnn",
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
        "n_train_prefixes": len(train_prefixes),
        "n_val_prefixes": len(val_prefixes),
        "n_test_prefixes": len(test_prefixes),
        "epochs_run": len(train_losses),
        "final_train_loss": train_losses[-1],
        "best_val_accuracy": float(best_val_acc),
        "overall_test_metrics": overall,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[{dataset_name}] wrote {result_dir} (checkpoint, per-k metrics, manifest.json)")


if __name__ == "__main__":
    main()
