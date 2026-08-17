# Phase 4 — Representation Extraction

Produced per [`PLAN.md`](../PLAN.md) Phase 4 / spec §7 ("For every trained model and every test case, extract a representation for every prefix... store per-trace Z_σ alongside true/predicted events, prediction probabilities, entropy, outcome, remaining time, prefix length, case metadata, correctness"). Builds on Phase 3's 9 trained checkpoints (`paper/phase3_baseline_reproduction.md`) — no retraining happens in this phase.

## Design

**Storage.** One parquet file per (dataset, model) at `results/<dataset>/<model>/embeddings_test.parquet`, colocated with that model's existing `checkpoint.pt`/`manifest.json`/`test_metrics_by_prefix_length.csv`, plus a companion `embeddings_manifest.json` for provenance. Long/tidy format — one row per (case_id, k) — rather than one row per whole trace: `Z_σ = (z_1,...,z_T)` is realized by grouping on `case_id` and sorting by `k`, the same convention every `test_metrics_by_prefix_length.csv` already uses.

**Schema.** Model-agnostic columns (`src/representations/common_fields.py`, computed once per run from the raw split + that model's own prefix table, independent of any architecture): `dataset`, `model`, `family`, `objective`, `case_id`, `k`, `prefix_length` (=k+1), `case_length`, `true_event`, `remaining_time_seconds`, `outcome`. Model-specific columns (computed by each `experiments/extract_<model>.py`): `z` (that model's own z_t, dimension varies by architecture — 36/100/128/32/80/512/200/64/64 for A1/A2/A3/A4/A5/A6/A7/B1/B2 respectively, intentionally never unified across models since geometric analysis only ever compares within one model's own latent space), `predicted_event`, `event_probs`, `pred_confidence`, `entropy`, `correct`, and — for the 5 full-suffix models (A4/A5/A6/A7/B2) only — `predicted_suffix`, `true_suffix`, `suffix_dl_similarity`.

**"Prediction probabilities"/entropy convention.** None of the 9 training scripts kept softmax probabilities anywhere — every `evaluate_*` function in Phase 3 discarded them immediately after `argmax`. This phase defines, uniformly across all 9 models and both objectives, "prediction probabilities" as the softmax distribution over the **immediate next event**: directly the model's own output for the 4 next-event models (A1/A2/A3/B1); the first generated/predicted suffix position's distribution for the 5 full-suffix models (A4/A5/A6/A7/B2) — computed as a genuine intermediate quantity each model's own `generate()`/direct-forward pass already produces internally, just not previously retained. This gives one consistent, comparable confidence/entropy definition everywhere, while `predicted_suffix`/`suffix_dl_similarity` separately preserve the richer full-suffix-specific signal (computed for free alongside `predicted_suffix`, reusing `evaluation.suffix_metrics.normalized_dl_similarity` — a small, deliberate superset of the spec's literal wording, since Phase 8/9/10 will need exactly this per-instance correctness signal and re-generating suffixes later would be wasteful). Entropy uses natural log (nats).

**`outcome` is null for every row, by explicit decision (not an oversight).** This project's canonical schema (`case_id`/`activity`/`timestamp` only) has no outcome label anywhere, and Phase 1 explicitly dropped outcome-prediction as a training objective for the whole roster (`STATUS.md`'s decision log — "H3 untested"; `configs/datasets/helpdesk.yaml` itself documents `outcome_label: null`). Asked explicitly during this phase whether to leave it null or define a post-hoc proxy label now: decided to leave it null, deferred to whichever later phase (6/8/9) actually needs a concrete outcome definition, rather than guess one now under Phase 4's narrower "representation extraction" scope.

**Cache versioning (`src/representations/cache.py`).** Each `embeddings_manifest.json` records a provenance dict hashing the source checkpoint's own bytes, the recomputed `dataset_split_hash` (hard-asserted equal to the training manifest's own recorded value, same pattern as `scripts/eval_sutran_val_dl.py`), and the extraction code itself (sha256 over the shared `src/representations/*.py` modules + that model's own extraction script). Every `experiments/extract_<model>.py` checks this against any existing manifest at startup and skips recomputation if nothing changed — verified empirically (see Results below): running all 9 twice in a row produces "cache up to date, skipping" on the second pass.

**Entrypoint convention.** One `experiments/extract_<model>.py` per model, mirroring `experiments/train_<model>.py`'s existing one-file-per-model convention. Each takes a dataset name only (default `helpdesk`) and reads everything else (`experiment_config`, `model_config`, `checkpoint_path`, `dataset_split_hash`) from that model's own `results/<dataset>/<model>/manifest.json` — the same "reload from manifest, no retrain" pattern already established by `scripts/eval_sutran_val_dl.py` — rather than from a possibly-since-edited `configs/experiments/*.yaml`. `scripts/extract_all_embeddings.py` runs all 9 in sequence for convenience.

## A real bug found and fixed along the way

While building `extract_process_transformer.py`, calling `build_vocab(parts["train"])` (mirroring `train_process_transformer.py`'s own call) crashed with `KeyError: 'history'`. Investigation (`git log`/`git show` on `src/models/process_transformer/adapter.py`) traced this to the A2 integration commit (`09ef0f4`, 2026-08-15 18:42): it refactored `build_vocab` to take a *prefix table* (needs a `history` column) instead of the raw train split, but **only updated `adapter.py` — the one call site in `train_process_transformer.py` was never updated**, leaving it passing the raw dataframe. Since A1 was trained *before* that commit (manifest `trained_at_utc`: 2026-08-15 15:45, three hours earlier), the checkpoint itself is unaffected, but **re-running `train_process_transformer.py` today would crash immediately** — a real, previously-undetected regression, not something introduced by this phase.

Before fixing anything, verified this wouldn't silently change A1's vocabulary if used for extraction: computed the vocabulary both ways (old: unique normalized activities directly from the train split; new: union of every prefix's `history`/`next_act` activities) against the real Helpdesk train split — identical 12-activity set, identical sorted order, so the two constructions are equivalent for this dataset (a length-1 case containing an activity that appears nowhere else would break this equivalence, but no such case exists here). Fixed both `train_process_transformer.py` (reordered so `build_vocab` receives `train_prefixes`, matching every other adapter's convention) and `extract_process_transformer.py` accordingly — confirmed the fix reproduces A1's exact recorded vocabulary and test accuracy (see Results).

## Results

All 9 caches extracted and every model's per-instance predictions cross-checked against its own Phase 3 training manifest — exact match (to displayed precision) in every case, strong end-to-end validation of the whole extraction pipeline across 4 frameworks (TensorFlow/Keras, PyTorch, PyTorch+DGL, PyTorch+HuggingFace) and every documented z_t extraction point:

| Model | n (test prefixes) | z_dim | Metric | Recorded (Phase 3) | Computed (Phase 4, per-instance mean) |
|---|---|---|---|---|---|
| A1 process_transformer | 3,495 | 36 | micro accuracy | 0.7293 | 0.7293 |
| A2 generative_lstm | 3,495 | 100 | micro accuracy | 0.6349 | 0.6349 |
| A3 rlhgnn | 3,495 | 128 | micro accuracy | 0.7210 | 0.7210 |
| A4 sutran | 4,411 | 32 | mean DL-similarity | 0.8164 | 0.8164 |
| A5 crtp_lstm | 4,411 | 80 | mean DL-similarity | 0.8158 | 0.8158 |
| A6 lupin | 4,411 | 512 | mean DL-similarity | 0.8288 | 0.8288 |
| A7 mlmme | 4,411 | 200 | mean DL-similarity | 0.8156 | 0.8156 |
| B1 controlled_transformer_next | 3,495 | 64 | micro accuracy | 0.6286 | 0.6286 |
| B2 controlled_transformer_suffix | 4,411 | 64 | mean DL-similarity | 0.8356 | 0.8356 |

Note the two different row counts across the roster (3,495 vs. 4,411 test prefixes) — a real, disclosed property of the two objective definitions, not a bug: `make_next_activity_prefixes` never emits a row for a case's final ("no next event") position, while `make_suffix_prefixes` does (a terminal row whose suffix is `[EOS]` only). Every next-event model therefore has exactly `n_test_prefixes` fewer rows than every full-suffix model on the same dataset, and each objective family's `compute_common_fields` output correctly reflects this (the terminal row's `true_event` is null for full-suffix models — see `tests/test_representations.py`'s `test_common_fields_suffix_prefixes_include_terminal_row`).

The cache-versioning mechanism was verified functionally, not just unit-tested: running `experiments/extract_controlled_transformer_next.py helpdesk` a second time (no changes to checkpoint/split/code) printed `embeddings cache up to date, skipping` instead of recomputing.

## Per-model notes worth flagging for Phase 5

- **A6 LUPIN's z_dim (512)** is `prajjwal1/bert-medium`'s hidden size — far larger than any other roster model's z_t (next largest: A3/A7 at 128/200) — worth normalizing for when Phase 5 compares intrinsic dimensionality *within* each model's own space (never raw magnitude across models, per spec §14).
- **A4 SuTraN's first-step distribution** required replicating `generate()`'s own fixed-full-window decoder-input construction exactly (a `window_size`-shaped tensor with only position 0 set to the SOS token) rather than a shorter ad-hoc call — SuTraN's hand-rolled cross-attention mask-broadcasting (documented in `model.py`'s `generate()` docstring) requires the decoder's query length to equal the encoder's padded `window_size` exactly; a naively shorter decode call would crash with a mask-broadcast shape mismatch. B2's equivalent one-step call has no such constraint (`nn.MultiheadAttention`'s masks don't couple query/key length), confirming this is SuTraN-specific, not a general Transformer requirement.
- **A7 MLMME's "probabilities"** are a softmax applied to the decoder's ReLU'd pre-activations (not a true softmax output layer) — matches the documented paper/code discrepancy already found in Phase 3 (`model.py`: paper says Softmax, code applies ReLU before cross-entropy) and how `CrossEntropyLoss` already treated those values internally during training; not a new modeling assumption.
- **A3 RLHGNN and A6 LUPIN both produce z_t already as a single pooled-per-prefix vector** in one forward pass (whole-graph max-pool; whole-text-sequence `[CLS]`, respectively) — no length-indexed gather needed, unlike every sequential model in the roster (A1/A2/A4/A5/B1/B2, which read out a specific position from a per-position output).

## Artifacts produced

- `src/representations/{__init__.py, common_fields.py, cache.py}`
- `experiments/extract_{process_transformer, generative_lstm, rlhgnn, sutran, crtp_lstm, lupin, mlmme, controlled_transformer_next, controlled_transformer_suffix}.py`
- `scripts/extract_all_embeddings.py`
- `tests/test_representations.py` (common_fields synthetic-data tests, cache staleness-detection tests, one representative model smoke test)
- Bug fix: `experiments/train_process_transformer.py`'s `build_vocab` call-site ordering (see above)
- `results/<dataset>/<model>/{embeddings_test.parquet, embeddings_manifest.json}` for all 9 models (gitignored, regenerate via `experiments/extract_<model>.py` or `scripts/extract_all_embeddings.py`)
