#!/usr/bin/env python
"""One-off follow-up: compute AR-generation DL-similarity on the VALIDATION
set for an already-trained SuTraN checkpoint, so it's directly comparable
to the test-set metric already in manifest.json (both same metric, unlike
best_val_loss which is cross-entropy). Does not retrain - loads the saved
checkpoint only. Updates manifest.json in place with the new key."""
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from data.loaders import load_dataset  # noqa: E402
from data.prefixes import make_suffix_prefixes  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from models.sutran.adapter import build_vocab, get_window_size  # noqa: E402
from models.sutran.model import SuTraNActivityOnly  # noqa: E402
from train_sutran import DEVICE, evaluate_suffix_generation  # noqa: E402

import torch  # noqa: E402


def main():
    result_dir = REPO_ROOT / "results" / "helpdesk" / "sutran"
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

    dataset_config = yaml.safe_load(
        (REPO_ROOT / manifest["experiment_config"]["dataset_config"]).read_text(encoding="utf-8")
    )
    model_config = manifest["model_config"]

    df = load_dataset(REPO_ROOT / dataset_config["raw_path"], dataset_config["source"]["format"])
    split_cfg = dataset_config["split"]
    split = compute_split(df, split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"])
    parts = apply_split(df, split)
    assert split.split_hash == manifest["dataset_split_hash"], "split hash mismatch vs. training run"

    train_prefixes = make_suffix_prefixes(parts["train"])
    val_prefixes = make_suffix_prefixes(parts["val"])
    vocab = build_vocab(train_prefixes)
    window_size = get_window_size(train_prefixes)
    assert window_size == manifest["window_size"]

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
    model.load_state_dict(torch.load(REPO_ROOT / manifest["checkpoint_path"], map_location=DEVICE))

    per_k, overall_val_dl_similarity = evaluate_suffix_generation(model, val_prefixes, vocab, window_size)
    print(f"Validation-set mean normalized DL similarity: {overall_val_dl_similarity:.4f}")

    manifest["overall_val_mean_dl_similarity"] = float(overall_val_dl_similarity)
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    per_k.to_csv(result_dir / "val_metrics_by_prefix_length.csv", index=False)
    print("Updated manifest.json with overall_val_mean_dl_similarity.")


if __name__ == "__main__":
    main()
