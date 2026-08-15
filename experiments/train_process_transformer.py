#!/usr/bin/env python
"""Train & evaluate A1 ProcessTransformer (next-event) on one dataset.

Usage:
    uv run --extra tf python experiments/train_process_transformer.py configs/experiments/pt_helpdesk.yaml

Config-driven per this project's convention: everything the run needs
(dataset, hyperparameters, seed) comes from the experiment config, nothing
is hardcoded here. Writes checkpoint + per-k metrics + a provenance
manifest (git commit, software versions, dataset split hash, config) to
results/<dataset>/process_transformer/.

Deliberate deviations from the original repo (see
src/models/process_transformer/adapter.py's docstring for the first two):
  1. Vocabulary built from TRAIN split only, not train+test combined.
  2. This project's own 64/16/20 time-based split is used, not the
     original repo's own 80/20 split.
  3. Checkpoint selection monitors VALIDATION accuracy, not training
     accuracy (the original script's ModelCheckpoint monitors
     `sparse_categorical_accuracy` on the training set itself, which
     doesn't measure generalization).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from models.process_transformer.adapter import build_vocab, encode, get_max_case_length, make_prefixes  # noqa: E402
from models.process_transformer.model import get_next_activity_model  # noqa: E402


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def software_versions() -> dict:
    return {pkg: _safe_version(pkg) for pkg in ("tensorflow", "keras", "numpy", "pandas")}


def _safe_version(pkg: str) -> str:
    try:
        return version(pkg)
    except Exception:
        return "unknown"


def evaluate_by_prefix_length(model, test_prefixes: pd.DataFrame, vocab, max_case_length: int) -> pd.DataFrame:
    from sklearn import metrics as skmetrics

    rows = []
    for k, group in test_prefixes.groupby("k"):
        token_x, token_y = encode(group, vocab, max_case_length)
        y_pred = np.argmax(model.predict(token_x, verbose=0), axis=1)
        rows.append(
            {
                "k": int(k),
                "n": len(group),
                "accuracy": skmetrics.accuracy_score(token_y, y_pred),
                "f1_weighted": skmetrics.f1_score(token_y, y_pred, average="weighted", zero_division=0),
                "precision_weighted": skmetrics.precision_score(token_y, y_pred, average="weighted", zero_division=0),
                "recall_weighted": skmetrics.recall_score(token_y, y_pred, average="weighted", zero_division=0),
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
    import tensorflow as tf

    dataset_name = dataset_config["name"]
    print(f"[{dataset_name}] loading raw log and re-deriving this project's split...")
    df = load_dataset(REPO_ROOT / dataset_config["raw_path"], dataset_config["source"]["format"])
    split_cfg = dataset_config["split"]
    split = compute_split(df, split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"])
    parts = apply_split(df, split)

    print(f"[{dataset_name}] building vocab from TRAIN split only...")
    vocab = build_vocab(parts["train"])
    train_prefixes = make_prefixes(parts["train"])
    val_prefixes = make_prefixes(parts["val"])
    test_prefixes = make_prefixes(parts["test"])
    max_case_length = get_max_case_length(train_prefixes)
    print(
        f"[{dataset_name}] vocab_size={vocab.vocab_size} num_classes={vocab.num_classes} "
        f"max_case_length={max_case_length} train_prefixes={len(train_prefixes)} "
        f"val_prefixes={len(val_prefixes)} test_prefixes={len(test_prefixes)}"
    )

    train_x, train_y = encode(train_prefixes, vocab, max_case_length)
    val_x, val_y = encode(val_prefixes, vocab, max_case_length)

    model = get_next_activity_model(
        max_case_length=max_case_length,
        vocab_size=vocab.vocab_size,
        output_dim=vocab.num_classes,
        embed_dim=model_config["embed_dim"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(model_config["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    result_dir = REPO_ROOT / "results" / dataset_name / "process_transformer"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "checkpoint.weights.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_weights_only=True,
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            save_best_only=True,
        )
    ]

    print(f"[{dataset_name}] training for {model_config['epochs']} epochs...")
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=model_config["epochs"],
        batch_size=model_config["batch_size"],
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
    )

    print(f"[{dataset_name}] restoring best (val-accuracy) checkpoint for test evaluation...")
    model.load_weights(str(checkpoint_path))

    per_k = evaluate_by_prefix_length(model, test_prefixes, vocab, max_case_length)
    per_k.to_csv(result_dir / "test_metrics_by_prefix_length.csv", index=False)

    # Two different "overall" aggregates, both reported since they can
    # legitimately differ: macro (mean across k-groups, comparable to the
    # original repo's reporting style) vs micro (global accuracy across all
    # test examples, standard and less sensitive to how many k-groups exist).
    test_x, test_y = encode(test_prefixes, vocab, max_case_length)
    y_pred_all = np.argmax(model.predict(test_x, verbose=0), axis=1)
    from sklearn import metrics as skmetrics

    overall = {
        "macro_avg_accuracy_across_k": float(per_k["accuracy"].mean()),
        "macro_avg_f1_weighted_across_k": float(per_k["f1_weighted"].mean()),
        "micro_accuracy": float(skmetrics.accuracy_score(test_y, y_pred_all)),
        "micro_f1_weighted": float(skmetrics.f1_score(test_y, y_pred_all, average="weighted", zero_division=0)),
    }
    print(f"[{dataset_name}] overall test metrics: {overall}")

    manifest = {
        "model": "process_transformer",
        "dataset": dataset_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "software_versions": software_versions(),
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
        "final_train_loss": float(history.history["loss"][-1]),
        "best_val_accuracy": float(max(history.history["val_sparse_categorical_accuracy"])),
        "overall_test_metrics": overall,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[{dataset_name}] wrote {result_dir} (checkpoint, per-k metrics, manifest.json)")


if __name__ == "__main__":
    main()
