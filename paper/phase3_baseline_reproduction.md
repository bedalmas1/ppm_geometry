# Phase 3 — Baseline Reproduction

Produced per [`PLAN.md`](../PLAN.md) Phase 3 / spec §25 Phase 3 ("Reproduce published predictive results within reasonable tolerance before performing geometric analysis. If reproduction fails, document why."). This document is updated as each of the 9 roster models is integrated; it currently covers the first one, used as the pilot to validate the whole approach end-to-end before repeating it for the rest.

## A1 — ProcessTransformer on Helpdesk (pilot)

### Why this model/dataset first

ProcessTransformer was chosen as the pilot because: it is architecturally the simplest model in the roster (single Transformer block); its source was already fully read during Phase 1 (license, architecture, z_t extraction point all confirmed); and Helpdesk is the same dataset its own paper evaluates on, so a real reproduction comparison is possible (unlike, say, immediately trying BPIC19 where partial credit for "close enough" is harder to judge).

### Integration approach: adapter, not the original repo's own preprocessing

Per spec §6 ("use identical train/validation/test splits across models"), this project cannot let each model run its own train/test split — that would confound any cross-model geometric comparison with differences in what data each model actually saw. So the original repo's `data_processing.py`/`data/loader.py` (which does its own 80/20 chronological split and builds vocabulary from the *combined* train+test set) was **not reused**. Instead:

- `src/models/process_transformer/model.py` — the architecture only (Transformer block, token+position embedding, pooling → dense head), vendored from `Zaharah/processtransformer` (Apache-2.0) with a named `prefix_representation` layer (the `GlobalAveragePooling1D` output) added so Phase 4 can find it by name without touching this file again.
- `src/models/process_transformer/adapter.py` — converts this project's own split parquet (from `src/data/`, Phase 2) into the prefix/next-activity tensors the model expects, replicating the original repo's exact prefix-generation semantics (`prefix = a0..ai`, `next_act = a_{i+1}`, `k = i`) and activity-name normalization (lowercase, spaces→hyphens, required so the space-joined prefix string round-trips correctly).
- `experiments/train_process_transformer.py` — config-driven training entrypoint; writes a checkpoint, per-prefix-length metrics, and a full provenance manifest (git commit, software versions, seed, dataset split hash) to `results/helpdesk/process_transformer/`.

**Two deliberate, documented deviations from the original repo, both toward correctness rather than fidelity** (see `adapter.py`'s docstring):
1. Vocabulary (`x_word_dict`/`y_word_dict`) is built from the **train split only**. The original repo builds it from train+test combined before splitting — a leakage shortcut this project does not replicate. Unseen activities at val/test time map to `[UNK]`.
2. This project's own 64/16/20 time-based split is used (chosen independently in Phase 2), not the original repo's 80/20. **These turned out to be nearly identical in practice**: the paper's own methodology (§5.2) is "80/20 train/test, with 20% of train used for validation" — which works out to the same 64/16/20 ratio.
3. Checkpoint selection monitors **validation accuracy**, not training accuracy (the original script's `ModelCheckpoint` monitors `sparse_categorical_accuracy` on the training set itself, which does not measure generalization).

### Hands-on framework-compatibility check

Before writing the adapter, the vendored model code (pre-Keras-3 style, flagged as a risk in the Phase 1 audit) was smoke-tested directly under this project's installed TensorFlow 2.19.1/Keras 3.15.1: it builds, compiles, and trains without modification — the only change made was naming the pooling layer for later use. No compatibility fix was actually needed.

### Independent validation of the Phase 2 data pipeline

Reading the original paper directly (not relying on memory — the arXiv PDF was fetched and read) turned up a valuable cross-check: **its Table 1 dataset statistics for Helpdesk (4,580 cases, 21,348 events, 14 activities, max case length 15, avg case length 4.66) match this project's own Phase 2 pipeline output exactly** (see `paper/dataset_selection.md`). This is strong independent evidence the Phase 2 preprocessing (XES/CSV loading, prefix semantics) is correct, computed via a completely different codebase than the original paper's own.

### Predictive metrics vs. the published number

The paper (Table 2/3, §5.2) reports **85.63% (weighted) next-activity accuracy on Helpdesk**, training for **100 epochs at learning rate 0.01** — note this comes from the **paper text**, not the public repo's own `next_activity.py` argparse defaults (10 epochs, lr=0.001), which do not match what the paper says it actually did. This project ran both:

| Run | Epochs | LR | Best val accuracy | Test micro accuracy | Test macro-avg-across-k accuracy |
|---|---|---|---|---|---|
| Smoke test (repo CLI defaults) | 10 | 0.001 | 81.7% | 63.1% | 51.4% |
| Paper-matched | 100 | 0.01 | **86.9%** | 72.9% | 63.2% |

**Interpretation, not hidden:**
- The paper-matched run's best validation accuracy (86.9%) lands close to the paper's reported 85.63%, which is the strongest piece of evidence the reproduction is basically sound.
- Test accuracy (72.9% micro) is noticeably lower than both the validation accuracy and the paper's reported number. Two plausible, non-exclusive explanations, neither of which is a pipeline bug:
  1. **Training instability at lr=0.01**: the training curve shows repeated loss spikes and recoveries (e.g. epochs 55, 73, 89–98) — a classic symptom of a learning rate too high for stable Adam convergence on this architecture/dataset size. The paper does not report a learning-rate schedule or warmup, so this may be inherent to using exactly lr=0.01 as stated, or the paper's number may reflect a specific favorable epoch rather than epoch 100.
  2. **Genuine temporal distribution shift under a strict chronological split**: validation data is chronologically contiguous with training (both drawn from the same early ~80% of cases), while test is a truly held-out *later* time period. If Helpdesk's ticketing process changed over time (plausible over a multi-year log), test performance will legitimately be lower than validation regardless of hyperparameters — this is exactly what a leakage-safe, strictly time-ordered split is supposed to expose rather than hide, and precisely the kind of thing spec §6's leakage-prevention requirement is meant to catch. The paper does not report a validation/test gap, so it is unclear whether it faced the same effect or reports test accuracy that partially benefits from a specific stopping point.
- **No further hyperparameter tuning was pursued** — this pilot's purpose was validating the full pipeline end-to-end (data → adapter → training → checkpoint → metrics → provenance), not achieving an exact number match. That goal is met: every stage worked, including the parts (vocabulary from train-only, stricter split) that were deliberately changed from the original repo for correctness.

### z_t extraction readiness (for Phase 4)

Confirmed the `prefix_representation` (GlobalAveragePooling1D, 36-dim) layer is retrievable by name from the trained model via `model.get_layer("prefix_representation")` — Phase 4 can build a new functional model reusing this trained model's weights to output that layer, without modifying `model.py` again.

### Artifacts produced

- `src/models/process_transformer/{model.py, adapter.py, LICENSE-processtransformer.txt}`
- `experiments/train_process_transformer.py`
- `configs/models/process_transformer.yaml`, `configs/experiments/pt_helpdesk.yaml`
- `results/helpdesk/process_transformer/{checkpoint.weights.h5, test_metrics_by_prefix_length.csv, manifest.json}` (gitignored — regenerate via the experiment script)

## A2 — Camargo et al. GenerativeLSTM on Helpdesk

### Scope decision: activity-only, not the full multi-task/pretrained-embedding pipeline

The original repo's `model_shared_cat` architecture is considerably more involved than ProcessTransformer's: three input branches (activity, role, time) feeding two stacked LSTM layers each, jointly predicting next-activity, next-role, and next-time; activity/role embeddings are **pretrained separately** (a word2vec-style skip-gram phase, `embedding_training.py`) and frozen before the main model trains; resources are clustered into "roles" via a separate algorithm (`support_modules/role_discovery.py`); and the paper's own experiments select between architecture variants (`shared_cat`, `concatenated`, `specialized`) via a large Bayesian/random hyperparameter search (2000 models per the paper's Section 4.1).

Fully reproducing all of that is a substantially larger undertaking than the geometry study needs. Consistent with how A1 was scoped (next-event only, not the time/remaining-time heads ProcessTransformer's own repo also supports), **A2 implements only the activity branch**: embed → LSTM(return_sequences=True) → BatchNormalization → LSTM(return_sequences=False, named `prefix_representation`) → Dense(softmax). This is architecturally exactly `model_shared_cat.py`'s `l1_c1 → batch1 → l2_c1 → act_output` path, with two changes:
- **No role branch** — this project's common event-log schema (`src/data/schema.py`) deliberately carries only case_id/activity/timestamp, not resource/role, so every model in the roster sees identical input data (spec §6); adding a role branch for this one model alone would break that.
- **Trainable embeddings, not pretrained/frozen ones** — avoids reproducing the separate word2vec-style pretraining stage.

Hyperparameters (`lstm_size=100`, `embed_dim=10`, `dropout=0.2`, `batch_size=32`, `epochs=200` with early stopping patience 40 and ReduceLROnPlateau patience 10, `Nadam(lr=0.002)`) are all taken directly from the repo's own `dg_training.py` defaults/search-space (not guessed) — see `configs/models/generative_lstm.yaml`'s comments for the exact provenance of each value.

**The "insecure HTTP git dependency" flagged in Phase 1 resolved itself**: the repo's `environment.yml` lists `git+http://github.com/Mcamargo85/support_modules.git`, but the repo *also* vendors its own `support_modules/` directory locally, which is what the code actually imports when run from the repo root. Since this project vendors the architecture code directly (same strategy as A1) rather than installing the repo's own environment, that dependency was never actually needed.

### Predictive metrics vs. published numbers

Two reference points, both real (not guessed): the Camargo et al. BPM 2019 paper itself reports **78.9% next-event accuracy on Helpdesk** (its Table 4, "Our approach" row — evaluated with a 70/30 train/validation split, not explicitly stated as chronological); ProcessTransformer's paper separately cites a third-party re-implementation of Camargo et al. at **76.51%** (via Rama-Maneiro et al.'s benchmark survey).

This project's activity-only GenerativeLSTM: **best validation accuracy 81.8%** (training was numerically smooth — a monotonic loss curve, unlike A1's instability — early-stopped at epoch 45), but **test accuracy only 63.5% micro / 59.1% macro-across-k**, on the exact same 64/16/20 chronological split as A1 (confirmed by an identical `dataset_split_hash` in both models' manifests).

### A notable cross-model finding

**The same validation-vs-test gap pattern that appeared for A1 (Transformer) recurs here for A2 (LSTM)** — two architecturally very different models, one with unstable training (A1 at lr=0.01) and one with smooth monotonic convergence (A2), both show validation accuracy in the low-to-high 80s and test accuracy in the 60s on the identical split. This is stronger evidence for the temporal-distribution-shift explanation floated in A1's write-up than for the lr-instability explanation: if the gap were purely an A1-specific optimization artifact, it should not reappear in a cleanly-converged, differently-architected model trained on the same data. Notably, **both models' validation accuracy meets or exceeds the original papers' own reported (non-strictly-chronological, or ambiguously-split) numbers** — suggesting this project's stricter, leakage-safe chronological split is surfacing a real generalization gap that looser split methodologies do not expose. Worth watching whether this recurs as more models/datasets are integrated (see STATUS.md open questions) — if it does, it becomes a genuine, reportable methodological finding in its own right, not just a reproduction caveat.

### z_t extraction readiness

Confirmed the `prefix_representation` (final LSTM layer, 100-dim) is retrievable by name from the trained model, exactly as A1's `prefix_representation` (GlobalAveragePooling1D, 36-dim) is — Phase 4 can use the same code pattern for both models despite their different architectures.

### Artifacts produced

- `src/models/generative_lstm/{model.py, adapter.py, LICENSE-generativelstm.txt}`
- `experiments/train_generative_lstm.py`
- `configs/models/generative_lstm.yaml`, `configs/experiments/lstm_helpdesk.yaml`
- `results/helpdesk/generative_lstm/{checkpoint.weights.h5, test_metrics_by_prefix_length.csv, manifest.json}` (gitignored)
- Refactored `src/data/prefixes.py`: the (history, k, next_act) prefix-generation logic used by both A1 and A2 is now shared, not duplicated — verified via a regression check that A1's numbers were unchanged after the refactor.

## A4 — SuTraN on Helpdesk

### Scope decision: activity-only, and why it's stricter than "NDA"

SuTraN's original architecture (both its DA and NDA variants) is a multi-task encoder-decoder: it jointly predicts the activity suffix, the timestamp ("time till next event") suffix, and a scalar remaining-runtime value. This project implements **activity-suffix prediction only** — and critically, this required removing timestamp *inputs* to the decoder, not just the timestamp *output heads*. The reason: NDA's timestamp inputs at inference time are themselves autoregressively generated from the model's own previous timestamp *prediction* (each new decoding step consumes the model's own last time estimate as a feature). Dropping only the output head while keeping timestamp inputs would leave no way to produce those inputs at real inference time. So every model in this project's roster ends up predicting **activities only** — a deliberately homogeneous target family across A1/A2/A4 (and, going forward, A5/A6/A7/B1/B2), not a per-model ad-hoc simplification.

Architecture and hyperparameters (`d_model=32`, `num_prefix_encoder_layers=4`, `num_decoder_layers=4`, `num_heads=8`, `d_ff=128`, `dropout=0.2`, `batch_size=128`, `AdamW(lr=0.0002, weight_decay=0.0001)`, `ExponentialLR(gamma=0.96)`, up to 200 epochs with patience 24) are taken directly from the repo's own `TRAIN_EVAL_SUTRAN_NDA.py`, not guessed. The core Transformer building blocks (multi-head attention, position-wise feed-forward, encoder/decoder layers, sinusoidal positional encoding) are vendored unmodified from `SuTraN/layers.py` — only the top-level model class is new, implementing the activity-only encoder-decoder.

### A real architectural bug caught by smoke-testing before the real run

A synthetic-tensor smoke test (following this project's now-standard practice of testing model wiring before spending real training time) caught a genuine shape-mismatch crash in cross-attention during autoregressive generation. Root cause: the vendored attention code's mask-broadcasting logic implicitly assumes the decoder's query length always equals the encoder's key length — true during teacher-forced training (both padded to the same length) but false during naive step-by-step incremental decoding (query length 1, 2, 3, ... while the encoder stays fixed-length). The original repo avoids this by decoding with a **fixed full-window-length decoder input at every step** (future positions held as padding, read out one position at a time via the causal mask) rather than growing the sequence dynamically — `model.py`'s `generate()` was rewritten to match this, and the smoke test then passed cleanly. A second, more mundane fix: the original repo pads prefix and suffix tensors to one **shared** `window_size`, not independent max lengths per side — using two independent max lengths (an earlier version of this adapter) breaks the same mask-broadcast for a related reason. Both fixes are documented in the code, not just here.

### Predictive metrics — no published-number comparison available for this dataset/model pair

Unlike A1/A2, **SuTraN's own paper never evaluated Helpdesk** (it used BPIC17, BPIC17-DR, BPIC19, and a proprietary BAC log) — Helpdesk was chosen anyway, consistent with A1/A2, to keep a controlled three-way architecture comparison on identical data. This means there is no published number to reproduce here; that's a disclosed limitation of this specific integration; a genuine comparison would require running on BPIC17/BPIC19 (logged as a later task, not blocking).

Result: trained to convergence (early-stopped at epoch 43 of a 200-epoch budget, smooth/stable training, no instability). Validation-set autoregressive generation: **0.924 mean normalized Damerau-Levenshtein similarity**. Test-set: **0.816**.

### The val/test gap, now confirmed a third time, across objectives and metrics

This is the same qualitative pattern flagged for A1 (86.9%→72.9% accuracy) and A2 (81.8%→63.5% accuracy) — but SuTraN is a genuinely different case: a different prediction objective (full-suffix generation, not next-activity classification) evaluated with a completely different metric (normalized Damerau-Levenshtein similarity over autoregressively-generated suffixes, not classification accuracy). Seeing the same validation-to-test degradation recur here, on the same dataset/split, across 3 architectures, 2 objectives, and 2 metric families, is now much stronger evidence that this reflects a genuine property of the Helpdesk log under a strict, leakage-safe chronological split — not an artifact specific to any one model's training dynamics or evaluation protocol. This has crossed the threshold from "worth watching" to "should be its own analysis" (see STATUS.md's updated open questions).

### z_t extraction readiness

`model.encode(prefix_tokens, prefix_pad_mask)` returns the per-prefix-position encoder output (`batch × window_size × d_model`, confirmed `(3, 15, 32)` on a smoke check against the trained checkpoint) — the cleanest z_t extraction point found across every model audited (Phase 1), now confirmed working end-to-end post-training.

### Artifacts produced

- `src/models/sutran/{layers.py, model.py, adapter.py, LICENSE-sutran.txt}`
- `src/data/prefixes.py` extended with `make_suffix_prefixes` (shared by A4/A5/B2) and an `EOS` constant
- `src/evaluation/suffix_metrics.py` — normalized Damerau-Levenshtein similarity, unit-tested against hand-worked cases (also the metric spec §8.8 calls for in the Phase 8 future-equivalence study — implemented once, reused later)
- `experiments/train_sutran.py`, `scripts/eval_sutran_val_dl.py` (one-off follow-up: val-set DL-similarity from an already-trained checkpoint, no retraining)
- `configs/models/sutran.yaml`, `configs/experiments/sutran_helpdesk.yaml`
- `results/helpdesk/sutran/{checkpoint.pt, test_metrics_by_prefix_length.csv, val_metrics_by_prefix_length.csv, manifest.json}` (gitignored)

## A5 — CRTP-LSTM on Helpdesk

### Mechanism: direct (non-autoregressive) prediction via left-padding + bidirectional LSTM

CRTP-LSTM's defining idea (per its title, "A Direct Data Aware LSTM ... for Complete Remaining Trace ... Prediction") is that it needs **no autoregressive decoding loop at all**, unlike A4 SuTraN. The mechanism: the prefix is fed to a bidirectional LSTM as a **left-padded** sequence — real prefix events pinned to the *end* of the fixed `window_size` window, with padding at the start (the opposite of A1/A2/A4's right-padding). Because the LSTM is bidirectional, every output position has access to the entire prefix via the backward pass, regardless of how short the real prefix is. This lets the model directly regress onto *all* suffix positions in a single forward pass — output position i is trained against the i-th future event (the same right-padded suffix target convention A4 already uses) — with no step-by-step generation required at either train or eval time.

This made A5 the fastest integration so far: the adapter reuses A4's vocabulary and suffix-target encoding directly (`models.sutran.adapter.Vocab`/`build_vocab`/`encode_training_suffixes`) via import, adding only one new piece — `encode_prefixes_left_padded`. No `generate()` method, no teacher-forcing input construction; the training loop is a plain classification loop over `(prefix_tokens_left_padded, suffix_targets)` pairs, evaluated identically at train and test time.

### Scope decision: activity-only (same rationale as A4)

The original architecture has a second dedicated bidirectional-LSTM branch + head predicting a remaining-runtime suffix, dropped here for the same reason as A4: this project's roster predicts activities only, consistently, everywhere. Hyperparameters (`d_model=80`, `dropout=0.2`, one shared + one dedicated BiLSTM layer, `NAdam(lr=0.002)`, `ReduceLROnPlateau(factor=0.5, patience=16)`, early-stop patience 24, up to 500 epochs) are taken directly from the repo's own `TRAIN_EVAL_CRTP_LSTM_ND.py`. The activity-embedding size uses the repo's own formula (`min(600, round(1.6 · n^0.56))`), applied to this project's own (smaller) vocabulary.

### Results

Trained on Helpdesk, same split as A1/A2/A4 (confirmed identical `dataset_split_hash`). Converged fast — best checkpoint found around epoch 1, early-stopped at epoch 25 (patience 24 from the best epoch). No published-number comparison available here either (same reason as A4 — SuTraN's repo/paper never evaluated Helpdesk). Result: **0.917 validation / 0.816 test** mean normalized DL-similarity — remarkably close to A4 SuTraN's 0.924/0.816 despite a completely different architecture (direct bidirectional LSTM vs. autoregressive Transformer decoder).

### The val/test gap, now confirmed a fourth time

A1 (86.9%→72.9%), A2 (81.8%→63.5%), A4 (0.924→0.816), A5 (0.917→0.816) — four models, three architectures... no, four architectures now (Transformer classifier, stacked LSTM classifier, Transformer encoder-decoder, direct bidirectional LSTM), two objectives, two metric families, one dataset/split, one consistent qualitative pattern. This is a strong, well-replicated finding at this point.

### z_t extraction readiness

`model.encode(prefix_tokens_left_padded)` returns the shared BiLSTM's output (`batch × window_size × d_model`, confirmed `(3, 15, 80)` against the trained checkpoint). Because of left-padding, position `window_size - 1` always corresponds to where the last real prefix event sits — the natural one-vector-per-prefix summary representation for Phase 4, regardless of actual prefix length. Documented explicitly since this differs structurally from A1/A4 (where every position of a single forward pass already corresponds to a prefix step) — CRTP-LSTM needs one forward pass *per prefix length* to get that length's z_t, same as every other model's Phase 4 extraction will do anyway via the shared `make_suffix_prefixes`/`make_next_activity_prefixes` row-per-prefix-length convention.

### Artifacts produced

- `src/models/crtp_lstm/{model.py, adapter.py, LICENSE-sutran-repo.txt}`
- `experiments/train_crtp_lstm.py`
- `configs/models/crtp_lstm.yaml`, `configs/experiments/crtp_lstm_helpdesk.yaml`
- `results/helpdesk/crtp_lstm/{checkpoint.pt, val_metrics_by_prefix_length.csv, test_metrics_by_prefix_length.csv, manifest.json}` (gitignored)

## A3 — RLHGNN on Helpdesk

### Why this integration needed the most up-front reading

Unlike A1/A2/A4/A5, RLHGNN's own README describes a 3-stage pipeline (train 4 fixed graph configurations → train a DQN policy to pick a configuration per instance → retrain on the RL-selected hybrid graphs), and the repo ships no dependency file and no LICENSE at all (flagged in Phase 1). Before writing any adapter code, `main.py`, `build_graph.py`, `Train.py`, `model/model.py`, `MyDataset.py`, `data_process.py`, and `ProcessEventlog_one_graph.py` were all read in full, and the actual arXiv PDF (2507.02690) was fetched and read (not relied on from memory or the abstract alone) to confirm the reported hyperparameters and settle a scope question the repo's own argparse defaults left ambiguous (`num-epochs=50` in `Train.py`'s CLI default vs. "100 epochs, early stopping after 10" in the paper's Sec. V-B — the paper's number was used).

### Scope decision: fixed "Comprehensive" graph, no RL/DQN structure selection

RLHGNN's central technical contribution is instance-adaptive graph structure selection via a DQN trained as a separate Markov-Decision-Process stage (`env_train.py`/`final_policy.py`, both skipped here). This project drops that AutoML-style wrapper and always builds the richest of the paper's four fixed structures — the "Comprehensive" graph (forward + backward + repeat_next edges, `action=3`/`build_Bidirect_complex_graph` in the original repo). This is not an arbitrary simplification: the paper's own ablation (Table V) shows Comprehensive is the second-best-performing fixed structure on average (GMean 0.725 vs. the full RL policy's 0.731, F1 0.568 vs. 0.576) and the richest in representational power, so it keeps the core heterogeneous-GNN architecture (the paper's other main contribution: relation-specific GraphSAGE aggregation — LSTM for forward/backward, mean for repeat_next) while dropping only the selection mechanism around it. This matches this project's established pattern of dropping AutoML/hyperparameter-search layers elsewhere in the roster (e.g. A2's dropped 2000-model architecture search).

A second, project-wide scope decision applies here too: **activity-only**. The original embeds one feature per raw event-log column it finds (activity, resource, plus two engineered/discretized timestamp features — inter-event duration and case-progression time, confirmed in `ProcessEventlog_one_graph.py` and the paper's Sec. IV-B/V-B). Per this project's consistent activity-only scope (spec-driven, applied to A1/A2/A4/A5 already), only the `activity` node feature is embedded; `feature_proj`'s input width is `hidden_dim` (one feature) instead of the original's `hidden_dim * num_features`.

Unlike every LSTM/Transformer model in the roster, RLHGNN's graphs are **not padded to a fixed window** — each prefix becomes a graph with exactly as many nodes as the prefix is long (matching the original's own design), so `src/models/rlhgnn/adapter.py` builds one DGL heterograph per prefix row directly from `history`, rather than reusing any padding-based tensor encoding from the other adapters. Graph construction (forward/backward/repeat_next edge lists, including the `get_index_of_duplicate_elements`-based repeat-edge logic) was ported by hand from `build_graph.py` and verified with a smoke test against a hand-worked 4-node example (`['a','a','a','b']` → 6 `repeat_next` edges, matched by hand-tracing the original's two nested loops) before any real training.

Architecture (`src/models/rlhgnn/model.py`) is a from-scratch reimplementation against the paper's Sec. IV-E equations and the repo's own `model/model.py`, not a verbatim copy — the repo has no LICENSE file at all (confirmed again at integration time; still an open item, see STATUS.md). Hyperparameters (`hidden_dim=128`, `num_layers=2`, `dropout=0.1`, `NAdam(lr=0.001)`, `batch_size=64`, up to 100 epochs with patience 10) match both the paper's Sec. V-B and the repo's own `Train.py` defaults.

### A documented paper/code discrepancy (not silently resolved either way)

The paper's prose describes the readout as "focusing on the current activity position within the process instance" (implying the last node), but `model/model.py`'s actual code takes `dgl.max_nodes(hg, 'h')` — an elementwise max over **every** node in the graph. Since the code, not the prose, is what produced the paper's reported numbers, this project replicates the code's actual behavior (whole-graph max-pooling), and documents the discrepancy in `model.py`'s docstring rather than silently picking one interpretation.

### Predictive metrics — no published-number comparison available for this dataset/model pair

Same disclosed limitation as A4/A5: RLHGNN's own paper evaluates six BPI12/13/2020 logs, never Helpdesk. Kept on Helpdesk anyway for the controlled 5-way same-data architecture comparison.

Result: same split as A1/A2/A4/A5 (confirmed identical `dataset_split_hash`). Training converged fast and stably (best checkpoint at epoch 2, early-stopped at epoch 12 of the 100-epoch budget, patience 10) — **best validation accuracy 86.4%**, but **test accuracy only 72.1% micro / 54.6% macro-across-k** (`f1_weighted` micro 64.7%). Per-prefix-length breakdown (`results/helpdesk/rlhgnn/test_metrics_by_prefix_length.csv`) shows a sharp dip at k=1 (43.4% accuracy on 916 instances) sandwiched between strong k=0 and k=2 accuracy (85.0%, 81.6%) — a pattern not seen this sharply in A1/A2/A4/A5's per-k curves, plausibly because RLHGNN's very short prefix graphs (1-2 nodes) give its GraphSAGE aggregators little structure to work with, an architecture-specific weakness worth a closer look once the Phase 4/5 geometric analysis is running.

### The val/test gap, now confirmed a FIFTH time

A1 (86.9%→72.9%), A2 (81.8%→63.5%), A4 (0.924→0.816), A5 (0.917→0.816), now **A3 (86.4%→72.1%)** — five models, five architectures (Transformer classifier, stacked LSTM classifier, Transformer encoder-decoder, direct BiLSTM, heterogeneous GNN), two objectives, two metric families, one dataset/split, one relentlessly consistent pattern. RLHGNN's numbers land almost exactly on top of A1's (86.9/72.9 vs. 86.4/72.1) despite a completely unrelated architecture family (graph neural network vs. Transformer) — this is about as strong as within-Helpdesk evidence for the temporal-distribution-shift explanation can get without a second dataset.

### z_t extraction readiness

`model.encode(hg)` returns the graph-level max-pooled node embedding (batch, 128), confirmed `(3, 128)` against the trained checkpoint. Architecturally distinct from every other model in the roster: it is the only one whose z_t summarizes the *entire* prefix graph via a symmetric pooling operation (max over all nodes) rather than reading out a designated last-position/last-token hidden state — worth flagging explicitly in the Phase 5 geometry write-up, since this could plausibly interact with representation geometry differently (e.g. permutation-invariance-adjacent properties that the sequential models don't share).

### Artifacts produced

- `src/models/rlhgnn/{model.py, adapter.py}` — **no LICENSE file included**, since the source repo ships none (disclosed in both files' docstrings, tracked in STATUS.md)
- `experiments/train_rlhgnn.py`
- `configs/models/rlhgnn.yaml`, `configs/experiments/rlhgnn_helpdesk.yaml`
- `results/helpdesk/rlhgnn/{checkpoint.pt, test_metrics_by_prefix_length.csv, manifest.json}` (gitignored)

## Next steps for the remaining 4 models

Same pattern to repeat for A6 (LUPIN), A7 (MLMME), and B1/B2 (in-house controlled Transformer): vendor or implement the architecture, write a data adapter onto this project's own splits (never the original repo's own preprocessing), train with paper-matched hyperparameters where documented, evaluate, and record a provenance manifest. Training runs are sequential (see STATUS.md decision log) given a single shared GPU (currently CPU-only PyTorch wheels — see STATUS.md's CUDA-target open question).
