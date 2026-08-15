#!/usr/bin/env python
"""Train & evaluate A2 GenerativeLSTM (next-event, activity-only scope) on
one dataset.

Usage:
    uv run --extra tf python experiments/train_generative_lstm.py configs/experiments/lstm_helpdesk.yaml

Mirrors experiments/train_process_transformer.py's structure and deliberate
deviations (train-only vocab, this project's own split, val-monitored
checkpointing) - see src/models/generative_lstm/model.py's docstring for the
scope decisions specific to this model (activity-only, trainable not
pretrained embeddings, fixed rather than searched hyperparameters).
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
from data.prefixes import make_next_activity_prefixes  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from models.generative_lstm.adapter import build_vocab, encode, get_max_case_length  # noqa: E402
from models.generative_lstm.model import get_next_activity_model  # noqa: E402


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


def _safe_version(pkg: str) -> str:
    try:
        return version(pkg)
    except Exception:
        return "unknown"


def software_versions() -> dict:
    return {pkg: _safe_version(pkg) for pkg in ("tensorflow", "keras", "numpy", "pandas")}


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
    train_prefixes = make_next_activity_prefixes(parts["train"])
    val_prefixes = make_next_activity_prefixes(parts["val"])
    test_prefixes = make_next_activity_prefixes(parts["test"])
    vocab = build_vocab(train_prefixes)
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
        lstm_size=model_config["lstm_size"],
        dropout=model_config["dropout"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=model_config["learning_rate"], beta_1=0.9, beta_2=0.999),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    result_dir = REPO_ROOT / "results" / dataset_name / "generative_lstm"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "checkpoint.weights.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_weights_only=True,
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=model_config["early_stopping_patience"]
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=model_config["reduce_lr_patience"],
            min_delta=0.0001,
        ),
    ]

    print(f"[{dataset_name}] training for up to {model_config['epochs']} epochs (early stopping patience={model_config['early_stopping_patience']})...")
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
        "model": "generative_lstm",
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
        "epochs_run": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "best_val_accuracy": float(max(history.history["val_sparse_categorical_accuracy"])),
        "overall_test_metrics": overall,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[{dataset_name}] wrote {result_dir} (checkpoint, per-k metrics, manifest.json)")


if __name__ == "__main__":
    main()
