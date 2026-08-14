# Phase 1 — Literature & Repository Audit

Produced per [`PLAN.md`](../PLAN.md) Phase 1 / spec §4, §25 Phase 1. Three parallel research passes covered: (1) next-event models, (2) full-suffix models, (3) outcome-prediction and self-supervised/representation-learning models, plus a check on the ICPM 2027 / ML4PM CFP. Findings below are as retrieved on 2026-08-14 via web search/fetch — repo activity dates, star counts, etc. should be re-verified at the point of actually cloning a repo in Phase 2/3, not trusted indefinitely from this snapshot.

**Note on reproducibility-status entries below:** "verified" means the source (README, `setup.py`, actual code file) was read; it does **not** mean the model was cloned and trained end-to-end. That confirmation is a Phase 3 task.

**Hands-on follow-up (2026-08-14):** the two Family A picks (ProcessTransformer, SuTraN) were actually `git clone`d and their source read directly (not just README/search-snippet review) to pin down license text and the exact z_t extraction point before committing to the roster below. Findings are folded into sections A/B and the roster section. Clones were scratch-only, not added to this repo.

---

## A. Next-event prediction candidates

| Paper | Year | Venue | Task | Architecture | Datasets (paper) | GitHub | License | Last update | Reproducibility status | Representation layer (pre-head) | Integration difficulty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ProcessTransformer** (Bukhsh, Saeed, Dijkman) | 2021 | arXiv 2104.00721 — formal peer-reviewed venue beyond arXiv **not confirmed** | Next-activity, next-time, remaining-time | Single-block self-attention Transformer (TF/Keras): `TokenAndPositionEmbedding` → one `TransformerBlock` (MHA + FFN, 2×LayerNorm) → `GlobalAveragePooling1D` → Dropout → Dense(64, relu) → Dropout → Dense(output_dim, linear) | 9 real-life logs (helpdesk, BPI variants, ...) | [Zaharah/processtransformer](https://github.com/Zaharah/processtransformer) | **Apache-2.0 — confirmed by reading `LICENSE.txt` directly (full Apache 2.0 header present)** | Last push 2024-09-26; still watched (49★/19 forks) | **Hands-on verified 2026-08-14**: repo cloned, `processtransformer/models/*.py` read in full. Confirmed single-block architecture exactly as described. `setup.py` declares deps with **no upper/lower version pins beyond `tensorflow>=2.4`** (`numpy`, `scikit-learn`, `pandas` fully unpinned) — a real reproducibility risk to fix in Phase 3 (pin exact versions per project conventions), not just a hypothetical one. Did not run a full TF install (heavy, and Phase 3 will set up the real training environment anyway). | Confirmed by reading source: the `GlobalAveragePooling1D` output (36-dim, task-agnostic pooled prefix representation, before any dense head layers) is the cleanest z_t; the subsequent `Dense(64, relu)` hidden layer is a second candidate for the Phase 10 "representation layer" ablation (spec §15) — this model conveniently gives us that ablation for free. | **S** |
| Tax et al., LSTM PPM | 2017 | CAiSE 2017 (peer-reviewed) | Next-activity+time, suffix, remaining time | Multi-task LSTM | 4 real-life logs | [verenich/ProcessSequencePrediction](https://github.com/verenich/ProcessSequencePrediction) | **None declared** | Last push 2019-06-18 | Stale, pre-TF2/Keras2, unclear licensing for reuse | Not inspected (rejected before deep inspection) | **L** — **rejected** |
| Camargo et al., GenerativeLSTM | 2019 | BPM 2019, LNCS 11675 (peer-reviewed) | Next-activity/time/role, suffix | LSTM + embeddings, extends Tax et al. | Real-life event logs | [AdaptiveBProcess/GenerativeLSTM](https://github.com/AdaptiveBProcess/GenerativeLSTM) | Apache-2.0 | Last push 2025-10-02 — actively maintained | Actively maintained, licensed; not run in this pass | Not yet pinned in source (Phase 2 task) — recommended over Tax's original repo | **M** |
| SPICE ("Towards Reproducibility in Predictive Process Mining") | Dec 2025 | arXiv 2512.16715 | Standardized PyTorch reimplementation of Tax/Camargo/ProcessTransformer, fixing leakage/BatchNorm/time-scaling bugs | PyTorch reimplementations | 11 datasets incl. BPI Challenges, Helpdesk | GitLab (Fraunhofer), not GitHub | **CC BY-NC-ND 4.0** — No-Derivatives clause blocks forking/modifying | Recent, described as active | Not usable as our codebase (ND license), but its documented bug list is a **must-read correctness reference** for Phase 2/3 preprocessing | N/A | **N/A — reference only, not a dependency** |
| "David vs. Goliath" (Weytjens & Weber) | Jun 2026 | arXiv 2606.15868, not yet peer-reviewed | Next-activity: argmax baseline vs. LSTM vs. Transformer vs. LLM (Qwen3) | Multiple, incl. HF GPT-2/Qwen3 | 4TU real-life logs | [hansweytjens/DavidGoliath](https://github.com/hansweytjens/DavidGoliath) | None declared | Repo created 2026-06-12, 0★ | Unvetted, days-old, no license | Not inspected | **watch item / citation only, not a baseline** |
| Käppel et al., "Attention Please: What Transformer Models Really Learn for Process Prediction" | 2024 | Springer BPM workshop chapter, arXiv 2408.07097 | Attention-based explainability for an unnamed next-activity Transformer | Unspecified | Unspecified | None found | CC BY-NC-ND | 2024-08 | No code — not a candidate model | N/A | **relevant only as related work for Phase 9** (attention-vs-geometry comparison) |

**Decision: ProcessTransformer is the Family A next-event pick.** Strongest available combination of permissive license, verified installability, and an unambiguous pre-head representation. Its exact venue provenance (arXiv only, no confirmed peer-reviewed proceedings) is a known weakness — noted rather than hidden, per the project's honesty requirement. No stronger, equally-reproducible alternative was found.

---

## B. Full-suffix prediction candidates

| Paper | Year | Venue | Task | Architecture | Datasets (paper) | GitHub | License | Last update | Reproducibility status | Representation layer (pre-head) | Integration difficulty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SuTraN** (Wuyts & De Weerdt) | 2024 | ICPM 2024 (peer-reviewed) | Suffix (activity+timestamp) + remaining runtime | PyTorch encoder-decoder Transformer, cross-attention; data-aware (DA) / non-data-aware (NDA) variants | BPIC17, BPIC17-DR, BPIC19, BAC (proprietary) | [BrechtWts/SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork) | **MIT — confirmed by reading `LICENSE` directly** | 16★/2 forks — small but real and dedicated | **Hands-on verified 2026-08-14**: repo cloned. Confirmed **PyTorch** (not TF) throughout — a useful framework contrast with ProcessTransformer (TF/Keras) and a data point for choosing Family B's implementation framework. `BPIC17_no_loop.zip`/`BPIC19.zip` data files and `TRAIN_EVAL_SUTRAN_DA.py`/`_NDA.py` entrypoints confirmed present in the repo root. No top-level `requirements.txt`/`environment.yml` — dependencies will need to be inferred from imports (`torch`, `sklearn`, `pandas`, `numpy`, `matplotlib`, `tqdm`) and pinned ourselves in Phase 3. | **Confirmed by reading `SuTraN/SuTraN.py`**: inside `forward()`, the prefix-event embeddings are passed through `self.encoder_layers` (a stack of `EncoderLayer` — default 3 layers) and the resulting tensor `x` (shape `batch × window_size × d_model`) is used directly as the cross-attention memory for the decoder, *before* any prediction head. This is an exact, per-prefix-position match for the spec's `z_t` trajectory definition — one `d_model`-dim vector per prefix length t, extracted in a single forward pass. This is the cleanest representation-extraction point found across every audited model. | **S** |
| SuTraN+ | 2025 | Follow-up repo by same authors; **peer-review status of SuTraN+ itself unconfirmed** | Suffix + runtime + **outcome**, jointly, adaptive multi-task weighting | Same encoder-decoder family, extended | BPIC17 (±loops), BPIC19, BAC | [BrechtWts/SuTraN_Plus](https://github.com/BrechtWts/SuTraN_Plus) | MIT | 5 commits — young repo | Real, MIT, same authors; treat as a research-repo extension, not a citable peer-reviewed baseline yet | Not documented in README excerpt (presumably shares SuTraN's encoder embedding + an outcome head) | **S/M — watch item**, see roster decision below |
| CRTP-LSTM (reimplemented inside the SuTraN repo) | prior work, reimplemented 2024 | original venue not independently verified here | Suffix prediction, LSTM encoder-decoder | LSTM encoder-decoder | Same as SuTraN repo | Same repo (`TRAIN_EVAL_CRTP_LSTM_DA.py`/`_ND.py`, module `CRTP_LSTM/`) | MIT (inherited) | same as SuTraN | Comes "for free" in the same codebase — **confirmed present** via clone (`CRTP_LSTM/CRTP_LSTM_model.py` etc.) | LSTM hidden state pre-head | **S — optional LSTM baseline, near-zero extra integration cost** |
| ED-LSTM and SEP-LSTM (also reimplemented inside the SuTraN repo, modules `LSTM_seq2seq/` and `OneStepAheadBenchmarks/`) | prior work, reimplemented 2024 | not independently verified | Suffix prediction, LSTM variants | Encoder-decoder LSTM / one-step-ahead LSTM | Same as SuTraN repo | Same repo (`TRAIN_EVAL_ED_LSTM.py`, `TRAIN_EVAL_SEP_LSTM.py`) | MIT (inherited) | same as SuTraN | **Confirmed present** via clone — two more LSTM-family suffix baselines available at zero extra integration cost if a broader architecture sweep is ever wanted | LSTM hidden state pre-head | **not adopted for the initial roster (see below), but noted for a future ablation if needed** |
| Uncertainty-Aware ED-LSTM for Probabilistic Suffix Prediction | 2025 | arXiv, peer-review unconfirmed | Suffix + uncertainty quantification | ED-LSTM | Not verified | Not located | — | — | Found via search snippet only, no repo verified | Unknown | **not shortlisted — repo unverified** |
| "Inter-case Informed Business Process Suffix Prediction" | 2026 | Springer (BPM-adjacent), exact venue unconfirmed | Suffix prediction using inter-case features | Unspecified | Not verified | Not located | — | — | Snippet only, no repo check | Unknown | **not shortlisted — repo unverified** |

**Decision: SuTraN remains the Family A full-suffix pick, as the spec's default suggestion.** It is still actively cited/built upon (SuTraN+ extends it; a 2026 inter-case suffix paper references the same line of work), so it has not been superseded. **CRTP-LSTM, already reimplemented in the same repo, is adopted as the optional LSTM baseline** (spec §4D) — this is a much lower-effort route to an LSTM comparison point than reviving Tax et al.'s stale original repo. **SuTraN+ is deliberately not adopted now** given its unconfirmed peer-review status; revisit only if Phase 2's hands-on inspection shows it's solid and the roster has room.

---

## C. Outcome-prediction candidates

| Paper | Year | Venue | Task | Architecture | Datasets | GitHub | License | Last update | Reproducibility status | Representation layer | Difficulty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Teinemaa et al., "Outcome-Oriented Predictive Process Monitoring" | 2019 | TKDD | Binary case outcome | RF / XGBoost / LR / SVM (classical ML, not neural) | 22 logs (8 BPIC/Sepsis/Hospital/Road-Traffic sources) | [irhete/predictive-monitoring-benchmark](https://github.com/irhete/predictive-monitoring-benchmark) | Apache-2.0 | Low recent activity (48 commits, 34★) | Structure suggests runnable, not independently executed | **None suitable** — tree/linear classifiers have no continuous prefix-trajectory representation | S to run, but **unusable for z_t extraction** |
| Stevens et al. (exact title/venue unconfirmed), explainable outcome prediction | ~2022–2023 | not confirmed | Binary case outcome | LSTM/CNN + attention-BiLSTM alongside classical ML | 13 logs (derived from Teinemaa benchmark) | [AlexanderPaulStevens/Explainability-in-Process-Outcome-Prediction](https://github.com/AlexanderPaulStevens/Explainability-in-Process-Outcome-Prediction) | Not confirmed from fetch | 102 commits, moderate activity | Partially verified; some notebooks are Colab-era, may need updating | LSTM hidden states / attention-BiLSTM pre-softmax representation, explicitly extractable | **M — too much integration risk (unconfirmed license) for the value added** |

**Decision: no dedicated Family A outcome model is adopted.** The verified outcome-prediction ecosystem is dominated by classical ML with no meaningful continuous z_t, and the one deep-learning candidate carries unconfirmed licensing and stale (Colab-era) tooling. **Outcome supervision is provided entirely by the project's own controlled Transformer (Family B)**, exactly as spec §4C anticipates ("outcome objective if labels can be defined meaningfully").

---

## D. Self-supervised / representation-learning candidates (optional, spec §4D)

| Paper | Year | Venue | Task | Architecture | Datasets | GitHub | License | Last update | Reproducibility status | Representation layer | Difficulty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Babaev et al., CoLES ("Contrastive Learning for Event Sequences with Self-Supervision") | 2022 | ACM SIGMOD | Self-supervised sequence embedding (event type + timestamp) | Transformer/RNN encoder + contrastive loss, via `pytorch-lifestream` | Financial transaction sequences (not BPM logs natively) | [dllllb/pytorch-lifestream](https://github.com/dllllb/pytorch-lifestream) | Not confirmed — check repo directly | Active, pip-installable, maintained demos | High confidence: actively maintained library | Pooled encoder output before contrastive projection head, natively exposed | **M — needs a process-mining data loader; core encoder is generic and reusable** |

No self-supervised method built and evaluated specifically on business-process event logs was found with a verifiable public repo in this pass (De Koninck 2018 / Guzzo 2021-style work is referenced in the literature but no repo was located).

**Decision: deferred, not adopted for the initial roster.** CoLES/`pytorch-lifestream` is the strongest candidate if a self-supervised control point is later judged necessary (e.g. if Phase 6's pilot suggests the descriptive study needs an unsupervised baseline for contrast), but adapting it to event-log data is nontrivial project work in itself, and every RQ in the spec is already addressed by the roster below without it. Revisit only if a specific gap emerges.

---

## E. ICPM 2027 / ML4PM workshop CFP

ICPM 2027 is confirmed for **Feb 8–12, 2027, University of Calabria** ([icpmconference.org/2027](https://icpmconference.org/2027/)). No ML4PM-specific workshop page, deadline, or page-limit was retrievable — **no concrete CFP found yet**, most likely because it is too early for the workshop program to be published. This must be re-checked closer to the writing phase (Phase 11) rather than assumed.

---

## Final model roster (finalized 2026-08-14)

6 configurations — at the upper end of the spec's 3–6 target, each independently justified; kept intentionally tight rather than padded (spec §26: strong baselines over many weak ones).

### Experiment Family A — ecological/SOTA comparison (published models, as-is)

| # | Model | Objective | Repo | License |
|---|---|---|---|---|
| A1 | ProcessTransformer | next-event | `Zaharah/processtransformer` | Apache-2.0 |
| A2 | SuTraN | full-suffix (+ remaining runtime) | `BrechtWts/SuffixTransformerNetwork` | MIT |
| A3 | CRTP-LSTM (from the SuTraN repo) | full-suffix, LSTM architecture | same repo as A2 | MIT |

### Experiment Family B — controlled comparison (one encoder, objective varied)

| # | Model | Objective | Implementation |
|---|---|---|---|
| B1 | Controlled Transformer | next-event | Built in-house (Phase 3), matched architecture/dim/budget across B1–B3 |
| B2 | Controlled Transformer | full-suffix | Built in-house |
| B3 | Controlled Transformer | outcome | Built in-house — this is where outcome supervision enters the study, per the Section C decision above |

Not adopted now, explicitly logged as deferred rather than silently dropped: SuTraN+ (outcome-capable SuTraN variant, Section B), CoLES/`pytorch-lifestream` (self-supervised baseline, Section D), Tax et al.'s original LSTM repo (superseded by Camargo's actively-maintained fork, though Camargo itself wasn't ultimately needed once CRTP-LSTM was available "for free" in the SuTraN repo — kept in this document for completeness, not part of the roster).

## Compute-estimate status

No candidate paper/repo reported concrete GPU-hour or training-time figures. Rather than guess, **Phase 2/3 should run one short empirical timing pilot** (a few epochs on the smallest selected dataset, for A1 and B1) to produce a real per-model/per-dataset compute estimate before committing to the full matrix — this was the one Phase 1 task (spec §25 point 7: "estimate computational requirements") that literature search could not satisfy and needs a hands-on check instead.
