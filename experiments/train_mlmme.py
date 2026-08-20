#!/usr/bin/env python
"""Train & evaluate A7 MLMME (full-suffix, activity-only, adversarially
trained LSTM encoder-decoder) on one dataset.

Usage:
    uv run --extra torch python experiments/train_mlmme.py configs/experiments/mlmme_helpdesk.yaml

See src/models/mlmme/model.py's module docstring for the full architecture/
training-procedure write-up (generator + discriminator, Gumbel-softmax
adversarial training, resolved paper-vs-code discrepancies, scope
decisions). Per-batch training alternates a discriminator update (fake
sequences detached, standard GAN practice) and a generator update
(minimizing the SUM of the adversarial "fool the discriminator" loss and
the standard supervised cross-entropy loss, exactly Algorithm 1 in the SDM
2021 paper) - no separate pretrain-then-adversarial-fine-tune phase, both
losses trained jointly from epoch 0. Evaluation uses hard-argmax greedy
(beam size 1) closed-loop generation (`generator.generate()`), matching the
paper's own Table 2 protocol and this project's other full-suffix models
(A4/A5/A6).

Checkpoint/resume: after every epoch, full training state (both models,
both optimizers, loss/timing history, and the random/numpy/torch RNG
states) is written to `<result_dir>/resume_state.pt`. Re-running this exact
command with the same result dir picks back up at the next epoch instead
of restarting - this project's execution environment kills long-running
background processes at ~60 minutes regardless of health (see
configs/models/mlmme.yaml's epochs comment), so a full 500-epoch run
(~18.75h at the measured ~135s/epoch) must span many separate invocations.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.prefixes import EOS, make_suffix_prefixes  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from evaluation.suffix_metrics import normalized_dl_similarity  # noqa: E402
from models.mlmme.adapter import (  # noqa: E402
    build_reverse_class_dict,
    build_vocab,
    encode_prefixes_classidx,
    encode_training_suffixes,
    get_window_size,
)
from models.mlmme.model import (  # noqa: E402
    Discriminator,
    MLMMEGenerator,
    anneal_gumbel_temperature,
    discriminator_adversarial_loss,
    generator_adversarial_loss,
    label_smooth_gumbel,
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


def masked_val_loss(generator, prefix_idx, lengths, target_idx, num_classes, batch_size=256) -> float:
    """Deterministic (teacher_forcing_ratio=0, fully self-fed) cross-entropy
    over the suffix, matching the original repo's own `model_eval_test`
    protocol for validation/test mode."""
    generator.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(prefix_idx), batch_size):
            p = prefix_idx[i : i + batch_size].to(DEVICE)
            lens = lengths[i : i + batch_size].to(DEVICE)
            t = target_idx[i : i + batch_size].to(DEVICE)
            prefix_onehot = F.one_hot(p, num_classes).float()
            suffix_onehot = F.one_hot(t, num_classes).float()
            y_pred = generator(prefix_onehot, lens, suffix_onehot, teacher_forcing_ratio=0.0)
            loss = F.cross_entropy(y_pred.reshape(-1, num_classes), t.reshape(-1), ignore_index=0)
            total += loss.item() * p.size(0)
            n += p.size(0)
    return total / n


def evaluate_generate(generator, prefix_df, vocab, prefix_idx, lengths, window_size, batch_size=256):
    """Hard-argmax greedy (beam size 1) closed-loop generation + normalized
    DL similarity per instance, grouped by prefix length k - see
    model.py's `generate()` docstring."""
    generator.eval()
    rev = build_reverse_class_dict(vocab)
    suffixes = prefix_df["suffix"].tolist()
    ks = prefix_df["k"].to_numpy()
    num_classes = vocab.num_classes
    all_sims, all_k = [], []

    with torch.no_grad():
        for i in range(0, len(prefix_df), batch_size):
            p = prefix_idx[i : i + batch_size].to(DEVICE)
            lens = lengths[i : i + batch_size].to(DEVICE)
            prefix_onehot = F.one_hot(p, num_classes).float()
            pred_classes = generator.generate(prefix_onehot, lens, max_len=window_size).cpu().numpy()

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
    parser.add_argument(
        "--max-epochs-override", type=int, default=None,
        help="Optional override for a timed smoke run (measure epoch time before committing to the full budget).",
    )
    parser.add_argument(
        "--result-dir", type=Path, default=None,
        help="Override the results output directory (defaults to results/<dataset>/mlmme).",
    )
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
    num_classes = vocab.num_classes
    print(
        f"[{dataset_name}] vocab_size={vocab.vocab_size} num_classes={num_classes} "
        f"window_size={window_size} train={len(train_prefixes)} val={len(val_prefixes)} test={len(test_prefixes)}"
    )

    train_prefix_idx, train_lengths = encode_prefixes_classidx(train_prefixes, vocab, window_size)
    _, train_target_idx = encode_training_suffixes(train_prefixes, vocab, window_size)
    val_prefix_idx, val_lengths = encode_prefixes_classidx(val_prefixes, vocab, window_size)
    _, val_target_idx = encode_training_suffixes(val_prefixes, vocab, window_size)
    test_prefix_idx, test_lengths = encode_prefixes_classidx(test_prefixes, vocab, window_size)

    generator = MLMMEGenerator(
        num_classes=num_classes,
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(DEVICE)
    discriminator = Discriminator(
        num_classes=num_classes,
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(DEVICE)

    optimizerG = torch.optim.RMSprop(generator.parameters(), lr=model_config["learning_rate"])
    optimizerD = torch.optim.RMSprop(discriminator.parameters(), lr=model_config["learning_rate"])

    teacher_forcing_ratio = model_config["teacher_forcing_ratio"]
    grad_clip_norm = model_config["grad_clip_norm"]
    gumbel_tau_smooth = model_config["gumbel_tau_smooth"]
    gumbel_tau_anneal_base = model_config["gumbel_tau_anneal_base"]
    gumbel_tau_floor = model_config["gumbel_tau_floor"]

    train_loader = DataLoader(
        TensorDataset(train_prefix_idx, train_lengths, train_target_idx),
        batch_size=model_config["batch_size"],
        shuffle=True,
    )

    result_dir = args.result_dir or (REPO_ROOT / "results" / dataset_name / "mlmme")
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "checkpoint.pt"
    resume_state_path = result_dir / "resume_state.pt"

    total_epochs = args.max_epochs_override or model_config["epochs"]

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_g_losses, train_d_losses, val_losses = [], [], []
    epoch_seconds = []
    stopped_early = False

    if resume_state_path.exists():
        print(f"[{dataset_name}] found {resume_state_path}, resuming training state...")
        resume_state = torch.load(resume_state_path, map_location=DEVICE, weights_only=False)
        generator.load_state_dict(resume_state["generator_state"])
        discriminator.load_state_dict(resume_state["discriminator_state"])
        optimizerG.load_state_dict(resume_state["optimizerG_state"])
        optimizerD.load_state_dict(resume_state["optimizerD_state"])
        start_epoch = resume_state["next_epoch"]
        best_val_loss = resume_state["best_val_loss"]
        epochs_without_improvement = resume_state["epochs_without_improvement"]
        train_g_losses = resume_state["train_g_losses"]
        train_d_losses = resume_state["train_d_losses"]
        val_losses = resume_state["val_losses"]
        epoch_seconds = resume_state["epoch_seconds"]
        stopped_early = resume_state["stopped_early"]
        random.setstate(resume_state["random_state"])
        np.random.set_state(resume_state["numpy_state"])
        torch.set_rng_state(resume_state["torch_state"].cpu())
        print(
            f"[{dataset_name}] resumed at epoch {start_epoch} "
            f"(best_val_loss={best_val_loss:.4f}, {len(train_g_losses)} epochs already run, "
            f"stopped_early={stopped_early})"
        )

    def save_resume_state(next_epoch: int, stopped_early_flag: bool) -> None:
        torch.save(
            {
                "generator_state": generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "optimizerG_state": optimizerG.state_dict(),
                "optimizerD_state": optimizerD.state_dict(),
                "next_epoch": next_epoch,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "train_g_losses": train_g_losses,
                "train_d_losses": train_d_losses,
                "val_losses": val_losses,
                "epoch_seconds": epoch_seconds,
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
    for epoch in range(start_epoch, total_epochs) if not stopped_early else []:
        t0 = time.time()
        generator.train()
        discriminator.train()
        epoch_g_loss, epoch_d_loss, n_batches = 0.0, 0.0, 0
        tau_anneal = anneal_gumbel_temperature(epoch, gumbel_tau_anneal_base, gumbel_tau_floor)

        for prefix_idx_b, lengths_b, target_idx_b in train_loader:
            prefix_idx_b = prefix_idx_b.to(DEVICE)
            lengths_b = lengths_b.to(DEVICE)
            target_idx_b = target_idx_b.to(DEVICE)
            prefix_onehot = F.one_hot(prefix_idx_b, num_classes).float()
            suffix_onehot = F.one_hot(target_idx_b, num_classes).float()

            # ---- adversarial pass: smoothed/relaxed inputs (see model.py docstring) ----
            smoothed_prefix = label_smooth_gumbel(prefix_onehot, tau=gumbel_tau_smooth)
            smoothed_suffix = label_smooth_gumbel(suffix_onehot, tau=gumbel_tau_smooth)
            y_pred_adv = generator(smoothed_prefix, lengths_b, smoothed_suffix, teacher_forcing_ratio)
            fake_seq = F.gumbel_softmax(y_pred_adv, tau=tau_anneal, dim=-1)
            real_seq = smoothed_suffix

            # Discriminator update (fake detached - standard GAN practice)
            optimizerD.zero_grad()
            real_scores = discriminator(real_seq)
            fake_scores_d = discriminator(fake_seq.detach())
            loss_d = discriminator_adversarial_loss(real_scores, fake_scores_d)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_norm)
            optimizerD.step()

            # ---- clean pass: MLE supervised loss ----
            y_pred_mle = generator(prefix_onehot, lengths_b, suffix_onehot, teacher_forcing_ratio)
            mle_loss = F.cross_entropy(
                y_pred_mle.reshape(-1, num_classes), target_idx_b.reshape(-1), ignore_index=0
            )

            # Generator update: adversarial (NOT detached, see model.py's
            # scope-decision docstring) + supervised loss, combined
            optimizerG.zero_grad()
            fake_scores_g = discriminator(fake_seq)
            loss_g_adv = generator_adversarial_loss(fake_scores_g)
            total_g_loss = loss_g_adv + mle_loss
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip_norm)
            optimizerG.step()

            epoch_g_loss += total_g_loss.item()
            epoch_d_loss += loss_d.item()
            n_batches += 1

        train_g_losses.append(epoch_g_loss / n_batches)
        train_d_losses.append(epoch_d_loss / n_batches)

        val_loss = masked_val_loss(generator, val_prefix_idx, val_lengths, val_target_idx, num_classes)
        val_losses.append(val_loss)
        epoch_seconds.append(time.time() - t0)
        print(
            f"Epoch {epoch}: train_g_loss={train_g_losses[-1]:.4f} train_d_loss={train_d_losses[-1]:.4f} "
            f"val_loss={val_loss:.4f} tau={tau_anneal:.4g} ({epoch_seconds[-1]:.1f}s)",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(generator.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= model_config["patience"]:
            print(f"No val_loss improvement for {model_config['patience']} epochs. Stopping at epoch {epoch}.")
            save_resume_state(next_epoch=epoch + 1, stopped_early_flag=True)
            break
        else:
            save_resume_state(next_epoch=epoch + 1, stopped_early_flag=False)

    print(f"[{dataset_name}] restoring best (val-loss) checkpoint for evaluation...")
    generator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    print(f"[{dataset_name}] running hard-argmax greedy (beam size 1) suffix generation on val + test sets...")
    val_per_k, overall_val_dl_similarity = evaluate_generate(
        generator, val_prefixes, vocab, val_prefix_idx, val_lengths, window_size
    )
    test_per_k, overall_test_dl_similarity = evaluate_generate(
        generator, test_prefixes, vocab, test_prefix_idx, test_lengths, window_size
    )
    val_per_k.to_csv(result_dir / "val_metrics_by_prefix_length.csv", index=False)
    test_per_k.to_csv(result_dir / "test_metrics_by_prefix_length.csv", index=False)
    print(f"[{dataset_name}] overall val mean normalized DL similarity: {overall_val_dl_similarity:.4f}")
    print(f"[{dataset_name}] overall test mean normalized DL similarity: {overall_test_dl_similarity:.4f}")

    manifest = {
        "model": "mlmme",
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
        "num_classes": num_classes,
        "window_size": window_size,
        "n_train_prefixes": len(train_prefixes),
        "n_val_prefixes": len(val_prefixes),
        "n_test_prefixes": len(test_prefixes),
        "epochs_run": len(train_g_losses),
        "mean_epoch_seconds": float(np.mean(epoch_seconds)),
        "final_train_g_loss": train_g_losses[-1],
        "final_train_d_loss": train_d_losses[-1],
        "best_val_loss": best_val_loss,
        "overall_val_mean_dl_similarity": float(overall_val_dl_similarity),
        "overall_test_mean_dl_similarity": float(overall_test_dl_similarity),
        "checkpoint_path": (
            str(checkpoint_path.relative_to(REPO_ROOT))
            if checkpoint_path.is_relative_to(REPO_ROOT)
            else str(checkpoint_path)
        ),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[{dataset_name}] wrote {result_dir} (checkpoint, per-k metrics, manifest.json)")


if __name__ == "__main__":
    main()
