# Phase 1 — Literature & Repository Audit

Produced per [`PLAN.md`](../PLAN.md) Phase 1 / spec §4, §25 Phase 1. This document went through three rounds:

1. **Initial pass** (2026-08-14): three parallel ad-hoc research passes covering next-event, full-suffix, and outcome-prediction/self-supervised models, plus a check on the ICPM 2027/ML4PM CFP.
2. **User-directed strengthening** (2026-08-14): the user judged a single model per Family A category too thin for reviewers; this added a second next-event model (Camargo's GenerativeLSTM) and resolved CRTP-LSTM's peer-review provenance, and dropped the outcome-prediction objective entirely.
3. **Systematic literature review** (2026-08-14): the user flagged that the ad-hoc approach had missed an entire architecture family (GNN-based next-event models), and asked for a properly systematic search — documented methodology, explicit inclusion/exclusion criteria — rather than sequential fork audits. This round covers 2021–2026, next-event and full-suffix only (outcome prediction stays dropped), and **relaxes the adoption bar from "peer-reviewed" to "peer-reviewed or arXiv preprint, provided a usable public repo exists"** per explicit user instruction, since the goal became "3–4 architecturally distinct models per category to run experiments on, decide paper inclusion later" rather than a minimal defensible pair.

**Note on reproducibility-status entries below:** "verified" means the source (README, `setup.py`, actual code file) was read; it does **not** mean the model was cloned and trained end-to-end. That confirmation is a Phase 3 task, except where a hands-on clone note below says otherwise.

**Hands-on clone verification performed to date** (license text + exact z_t extraction point read from source, not just README/snippet): ProcessTransformer, SuTraN, CRTP-LSTM (partial — architecture confirmed, exact layer TBD), Camargo GenerativeLSTM, LUPIN, MLMME, RLHGNN, TGN-AST. All clones were scratch-only, never added to this repo.

---

## A. Next-event prediction — candidates and final roster

### A.1 Systematic search methodology (2021–2026, non-GNN architectures)

Documented for reuse in the paper's related-work/model-selection section. Sources queried: web search acting as an aggregate proxy for Google Scholar / Semantic Scholar / DBLP / arXiv / ACM DL / IEEE Xplore / SpringerLink (no direct per-database API access was available — this is a real limitation, not an exhaustive multi-database query, and should be disclosed as such). Search terms: combinations of "predictive process monitoring" / "next activity prediction" / "next event prediction" / "business process prediction" with architecture terms (transformer, LSTM, GRU, attention, hybrid, mixture-of-experts, foundation model, LLM). Venues prioritized: ICPM, BPM, CAiSE, ER, ICSOC, Information Systems, Decision Support Systems, Expert Systems with Applications, Knowledge-Based Systems, Journal of Intelligent Information Systems, IEEE Transactions on Services Computing/Neural Networks/Knowledge and Data Engineering, Neurocomputing, plus arXiv-with-code. Date range 2021–2026. Inclusion: single-case XES-style next-activity prediction, any neural architecture (GNN searched separately, §A.2). Exclusion: OCEL/object-centric logs, classical ML with no continuous representation, survey/benchmark papers (listed separately).

### A.2 Systematic search methodology (GNN-based architectures, 2021–2026)

Separate dedicated pass, since this was the category the ad-hoc audits missed entirely. Same source/venue/date-range approach as §A.1, with terms broadened to graph neural network, GCN, GAT, graph transformer, heterogeneous graph, directly-follows graph (DFG). Considered architectures modeling: process/DFG structure, case-level graphs, activity co-occurrence graphs, heterogeneous graphs combining control-flow with case attributes. Explicitly excluded object-centric event log (OCEL) work — a different data model than this project's single-case XES logs.

### A.3 Candidate table

| Paper | Year | Venue | Peer review | Architecture | Repo | License | Reproducibility verdict |
|---|---|---|---|---|---|---|---|
| **ProcessTransformer** (Bukhsh, Saeed, Dijkman) | 2021 | arXiv 2104.00721 | Not confirmed beyond arXiv (disclosed weakness) | Single-block Transformer, TF/Keras | [Zaharah/processtransformer](https://github.com/Zaharah/processtransformer) | Apache-2.0 (confirmed, `LICENSE.txt` read) | **Adopted.** Hands-on verified: single `TransformerBlock` → `GlobalAveragePooling1D` → Dense heads. z_t = pooled output before the Dense(64) hidden layer (which is itself a ready-made 2nd choice for the Phase 10 layer ablation). `setup.py` unpinned beyond `tensorflow>=2.4` — fix in Phase 3. |
| **Camargo et al., GenerativeLSTM** | 2019 | BPM 2019, LNCS 11675 | Confirmed peer-reviewed | Stacked LSTM (+GRU variants), TF/Keras — **same framework as ProcessTransformer**, simplifying environment planning | [AdaptiveBProcess/GenerativeLSTM](https://github.com/AdaptiveBProcess/GenerativeLSTM) | Apache-2.0 (confirmed, `LICENSE` read) | **Adopted.** Hands-on verified: `model_training/models/model_shared_cat.py` (and 9 other architecture variants in the same folder) build a 2-layer stacked LSTM per input branch, merged before Dense output heads — z_t = final LSTM layer output pre-merge. **New finding:** `environment.yml` pulls an unpinned dependency over **plain HTTP**: `git+http://github.com/Mcamargo85/support_modules.git` — a real reproducibility/security risk, not just a style issue; pin to a commit and use HTTPS in Phase 3. |
| **RLHGNN** (Wang, Yu, Song, Cao, Fan, Zhang) | 2025 | arXiv 2507.02690, not peer-reviewed | Not peer-reviewed | Heterogeneous graph + RL-driven graph construction, PyTorch + **DGL** (Deep Graph Library) | [Joker3993/RLHGNN](https://github.com/Joker3993/RLHGNN) | **None — confirmed absent, no LICENSE file in the repo at all** | **Adopted, with a disclosed risk.** Hands-on verified: `model/model.py`'s `HeteroSAGE.forward()` builds node embeddings via heterogeneous graph-conv layers (`self.hetero_convs`), then `graph_embed = dgl.max_nodes(hg, 'h')` before `self.classifier` — a clean pre-head z_t. This is the only candidate across both audits (this pass and the earlier GNN-specific one) that delivers genuine GNN architectural diversity with actual working code targeting single-case next-activity prediction (not OCEL). **The missing license is a real gap**, not a formality: no other model in this roster has an unlicensed repo. Recommendation: use for research/benchmarking with clear attribution, disclose the license gap explicitly in the paper, and consider emailing the authors for clarification before any wider redistribution of derived code. No `requirements.txt` either — dependencies (`torch`, `dgl`, `sklearn`) must be pinned ourselves. |
| ~~TGN-AST~~ (Hennig & Schmidt) | 2025 | BPM 2025 main proceedings, LNCS 16044 | Confirmed peer-reviewed (stronger venue than ProcessTransformer) | **Corrected on hands-on inspection**: NOT an end-to-end GNN. It is a **TensorFlow/Keras** hierarchical attribute-selection Transformer (`BaseHierarchicalProcessTransformer`, `MultitaskHierarchicalProcessTransformer` classes read directly from the notebook) that consumes, as one input feature, temporal node embeddings *pretrained separately* by a temporal graph network over organizational/resource relations. The graph component is a decoupled preprocessing step, not part of the differentiable predictive model. | [mchennig/tgn-ast](https://github.com/mchennig/tgn-ast) | MIT (confirmed) | **Rejected, not adopted.** Despite a stronger peer-review venue than ProcessTransformer and a real license, it (a) does not deliver the GNN architectural diversity it was sought for — it's a Transformer-family model much like ProcessTransformer, just with an extra engineered feature; (b) is Colab-oriented and notebook-only (no scripted entrypoints, README explicitly warns "adjustments of local variables might be necessary"); (c) is a 642MB repo with bundled dataset zips. The integration cost clearly exceeds the value given (a). |
| Nguyen et al., Switch-Transformer PPM | 2024 | Int'l Conf. Advances in AI (ACM), peer-reviewed | Confirmed | Switch-Transformer (mixture-of-experts) | Not found | — | **Not reproducible — no located repo** |
| Jalayer et al., HAM-Net | 2022 | Knowledge-Based Systems 236 | Confirmed | Hierarchical attention | Not found | — | **Not reproducible — no located repo** |
| Wickramanayake et al., shared/specialised attention LSTM | 2022 | Knowledge-Based Systems 248 | Confirmed | Attention-based LSTM (2 variants) | Not found | — | **Not reproducible — no located repo** |
| Fertig, Kirchdorfer, Sesterhenn, foundation-model PPM comparison | 2026 | ECML PKDD 2026 Workshops (accepted, not yet published) | Pending | Sequence models vs. tabular foundation models vs. LLMs | Not found | — | **Not reproducible — no repo, not yet published** |
| "Exploring LLM Features in PPM for Small-Scale Event-Logs" | 2026 | arXiv 2601.11468 / Springer chapter | Springer chapter suggests peer review, unconfirmed | LLM-feature-based | Not found | — | **Not reproducible — no located repo** |
| Weinzierl, GGSNN for next activity | 2021 | ICPM 2021 Workshops, LNBIP 433 | Confirmed | Instance graph per case, Gated Graph Sequence NN | Not found | — | **Not reproducible — no located repo** |
| Chiorrini et al., multi-perspective instance graphs | 2023 | Journal of Intelligent Information Systems | Confirmed | Deep Graph Convolutional NN | Not found | — | **Not reproducible — no located repo** |
| Dissegna & Di Francescomarino, GNN-for-PPM review/benchmark | 2025 | BPM 2024 Workshops, LNBIP 534 | Confirmed | Survey/benchmark, not a single model | Not found | — | **Not a candidate model — relevant as related-work citation for the survey itself; own reproducibility unverified (paywalled)** |
| Dissegna, Di Francescomarino, Ronzani, heterogeneous GNN | 2025 | RCIS 2025, LNBIP 547 | Confirmed | Heterogeneous GNN, multi-perspective | Not found | — | **Not reproducible — no located repo** |
| Lischka, Rauch, Stritzel, DFG-based GNN comparison | 2025 | arXiv 2503.03197 | Not peer-reviewed | GCN/GAT/GREAT over per-trace DFG | **Not found** (uses PyTorch Geometric, but no repo published) | — | **Not reproducible — preprint only, no code** |
| Wang & Damiani, Time-Aware/Transition-Semantic GNNs | 2025 | arXiv 2508.09527 / SSRN preprint | Not peer-reviewed | Prefix-based GCN + full-trace GAT, time-decay attention | [skyocean/TemporalAwareGNNs-NextEvent](https://github.com/skyocean/TemporalAwareGNNs-NextEvent) | Not checked (license unread) | **Not hands-on verified** — found with code, but not pursued once RLHGNN was already adopted as the roster's GNN representative; explicitly interpretability-focused rather than a general next-event predictor |
| EHHN, PROPHET (object-centric next-activity GNNs) | 2025/2026 | Not confirmed | Not confirmed | Heterogeneous hypergraph over **OCEL** logs | Not checked | — | **Out of scope — object-centric data model, not this project's single-case XES logs** |

### A.4 Headline finding

Across both the GNN-specific pass and the broader non-GNN systematic pass, **publication volume for next-event PPM models (2021–2026) is high but code availability is low**: of ~17 distinct papers surfaced, only 2 had any public repo at all beyond what was already known (TGN-AST, RLHGNN), and one of those (TGN-AST) turned out not to deliver the architecture family it appeared to. This directly informed the final decision: rather than force a wide next-event sweep, the roster keeps 3 models spanning 3 genuinely distinct architectures (Transformer / LSTM / heterogeneous GNN), and logs everything else found as a documented "no usable code" gap for a possible journal-extension follow-up (contacting authors, or reimplementing).

### A.5 Final next-event roster

**A1** ProcessTransformer (Transformer, TF/Keras) · **A2** Camargo GenerativeLSTM (LSTM, TF/Keras) · **A3** RLHGNN (heterogeneous GNN, PyTorch+DGL, license gap disclosed).

---

## B. Full-suffix prediction — candidates and final roster

### B.1 Systematic search methodology (2021–2026, all architectures)

Same source/venue approach as §A.1. Search terms: combinations of "predictive process monitoring" / "suffix prediction" / "remaining trace prediction" / "business process prediction" with architecture terms (transformer, LSTM, GRU, GNN, GAN, diffusion, hybrid, encoder-decoder, seq2seq). Date range 2021–2026 (one boundary exception: Taymouri & La Rosa 2021/SDM sits right at the edge and is CRTP-LSTM's direct lineage, included). Inclusion: full/multi-step suffix prediction (activity sequence, ideally + timestamps/remaining time) for single-case XES logs, any neural architecture. Exclusion: OCEL work, one-step-ahead models not evaluated as suffix predictors by their own paper, classical ML, survey/benchmark papers.

### B.2 Candidate table

| Paper | Year | Venue | Peer review | Architecture | Repo | License | Reproducibility verdict |
|---|---|---|---|---|---|---|---|
| **SuTraN** (Wuyts & De Weerdt) | 2024 | ICPM 2024 | Confirmed | PyTorch encoder-decoder Transformer, cross-attention, DA/NDA variants | [BrechtWts/SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork) | MIT (confirmed) | **Adopted.** z_t = `self.encoder_layers` output tensor (`batch × window_size × d_model`), used as decoder cross-attention memory before any head — cleanest extraction point found across every model audited. No top-level dependency file; pin `torch`/`sklearn`/`pandas`/`numpy` versions ourselves. |
| **CRTP-LSTM** (Gunnarsson, vanden Broucke & De Weerdt) | 2023 | IEEE Transactions on Services Computing 16(4), pp. 2330–2342 | Confirmed | Data-aware LSTM, direct/non-autoregressive full-remaining-trace prediction | Reimplemented in the SuTraN repo (`CRTP_LSTM/`) | MIT (inherited) | **Adopted.** Genuinely peer-reviewed (resolved from "unconfirmed" in the initial pass by a dedicated follow-up search) — an established IEEE journal, not a workshop note. Architecturally distinct from SuTraN (direct LSTM vs. autoregressive Transformer). Exact z_t layer still TBD (Phase 2/3 source read, same treatment as the others). |
| **LUPIN** (Pasquadibisceglie, Appice, Malerba) | 2024 | ICPM 2024 | Confirmed | **BERT fine-tuning** (HuggingFace `transformers`), ships Integrated-Gradients explainability | [vinspdb/LUPIN](https://github.com/vinspdb/LUPIN) | **CC BY-NC-SA 4.0 (confirmed, `LICENSE.md` read)** | **Adopted.** Hands-on verified: `neural_network/llamp_multiout.py`'s `BertMultiOutputClassificationHeads.forward()` returns `outputs.pooler_output` fed into per-task `nn.Linear` heads — z_t = BERT's pooler output, as clean an extraction point as ProcessTransformer's pooling layer. `requirements.txt` is **fully version-pinned** (best reproducibility hygiene of any model audited). Same venue/vintage as SuTraN (ICPM 2024), no recency concession needed. **License caveat**: CC BY-NC-SA 4.0 (non-commercial, share-alike) is fine for academic research use but its code must stay isolated/clearly attributed rather than merged into permissively-licensed code, the same handling SPICE's ND license got — do not vendor it into a differently-licensed shared module. |
| **MLMME** (Taymouri & La Rosa) | 2021 | SDM 2021 (SIAM) | Confirmed | RNN (LSTM) encoder-decoder + **GAN adversarial training** (Gumbel-Softmax) | [farbodtaymouri/MLMME](https://github.com/farbodtaymouri/MLMME) | GPL-3.0 (confirmed, `LICENSE` read) | **Adopted.** Hands-on verified: `network.py`'s `Encoder.forward()` returns `(h, c)` LSTM hidden/cell state — z_t = final encoder hidden state, pre-decoder. Official author repo, real entrypoint (`main.py`). Chosen over I3SP as the 4th suffix model because its adversarial-training paradigm is a genuinely distinct axis (training regime, not just architecture) versus CRTP-LSTM's supervised direct-LSTM — more architectural diversity than another LSTM-family encoder-decoder would add. **GPL-3.0 is copyleft** — stronger a constraint than any other license in the roster; keep its code isolated the same way as LUPIN's. |
| I3SP (Xiaomeng-He et al.) | 2025/2026 | BPM 2025 Workshops, LNBIP 569 | Confirmed, independent of the SuTraN/CRTP-LSTM (KU Leuven) group | Inter-case-aware Seq2Seq LSTM | [Xiaomeng-He/I3SP](https://github.com/Xiaomeng-He/I3SP) | MIT | **Deferred, not adopted.** Considered specifically to defuse a "SuTraN/CRTP-LSTM are the same lab/codebase" critique; not picked as the 4th model since MLMME's adversarial-training paradigm was judged to add more architectural diversity than another LSTM-family encoder-decoder. Logged as the fallback if the same-lab critique is raised in review. |
| Mustroph, Kunkler, Rinderle-Ma, uncertainty-aware suffix prediction | 2025 | IEEE conference (ICPM-adjacent) | Confirmed (resolves this paper's earlier "unconfirmed" status) | Uncertainty-aware ED-LSTM + Monte Carlo suffix sampling | Not located | — | **Not reproducible — no repo despite confirmed peer review** |
| Ali, Dumas, Milani, sweep-line suffix prediction | 2026 | CoopIS 2025, LNCS 15535 | Confirmed | Sweep-line, lockstep multi-case prediction | Not located | — | **Not reproducible — no repo** |
| "Hierarchical structuring of bilaterally expanding subtrace patterns" | 2026 | Process Science (Springer) | Confirmed | Tree-based (non-neural) | Not checked | — | **Excluded — non-neural, no continuous z_t** |
| Diffusion-model suffix prediction | — | — | — | — | — | — | **Searched explicitly, none found — a genuine negative result, not a gap in search effort** |
| ED-LSTM / SEP-LSTM (also in the SuTraN repo) | prior work, reimplemented 2024 | not independently verified | LSTM variants | [BrechtWts/SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork) (`LSTM_seq2seq/`, `OneStepAheadBenchmarks/`) | MIT (inherited) | **Not adopted** — available at zero extra cost if a broader LSTM sweep is ever wanted, but MLMME already covers the "adversarial LSTM" niche and CRTP-LSTM the "direct LSTM" niche |

### B.3 Headline finding

Full-suffix prediction is a **materially richer reproducible category than the initial ad-hoc pass suggested**: 4 models (SuTraN, CRTP-LSTM, LUPIN, MLMME) meet the peer-reviewed-or-arXiv + usable-repo bar, spanning 4 genuinely different architectural paradigms (autoregressive Transformer encoder-decoder / direct non-autoregressive LSTM / BERT-LLM fine-tuning / adversarially-trained RNN encoder-decoder). This contrasts sharply with next-event's near-total absence of code — the two categories are not symmetric in how much the literature has actually published alongside working implementations.

### B.4 Final full-suffix roster

**A4** SuTraN · **A5** CRTP-LSTM · **A6** LUPIN (license caveat: CC BY-NC-SA 4.0) · **A7** MLMME (license caveat: GPL-3.0).

---

## C. Outcome-prediction candidates

| Paper | Year | Venue | Task | Architecture | Datasets | GitHub | License | Last update | Reproducibility status | Representation layer | Difficulty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Teinemaa et al., "Outcome-Oriented Predictive Process Monitoring" | 2019 | TKDD | Binary case outcome | RF / XGBoost / LR / SVM (classical ML, not neural) | 22 logs (8 BPIC/Sepsis/Hospital/Road-Traffic sources) | [irhete/predictive-monitoring-benchmark](https://github.com/irhete/predictive-monitoring-benchmark) | Apache-2.0 | Low recent activity (48 commits, 34★) | Structure suggests runnable, not independently executed | **None suitable** — tree/linear classifiers have no continuous prefix-trajectory representation | S to run, but **unusable for z_t extraction** |
| Stevens et al. (exact title/venue unconfirmed), explainable outcome prediction | ~2022–2023 | not confirmed | Binary case outcome | LSTM/CNN + attention-BiLSTM alongside classical ML | 13 logs (derived from Teinemaa benchmark) | [AlexanderPaulStevens/Explainability-in-Process-Outcome-Prediction](https://github.com/AlexanderPaulStevens/Explainability-in-Process-Outcome-Prediction) | Not confirmed from fetch | 102 commits, moderate activity | Partially verified; some notebooks are Colab-era, may need updating | LSTM hidden states / attention-BiLSTM pre-softmax representation, explicitly extractable | **M — too much integration risk (unconfirmed license) for the value added** |

**Decision: outcome prediction is dropped from the study entirely** (not just from Family A — no Family B outcome model either). H3 is explicitly deferred/untested, stated plainly rather than hidden. Not re-opened during the systematic review round, per user instruction to scope that round to next-event + full-suffix only.

---

## D. Self-supervised / representation-learning candidates (optional, spec §4D)

| Paper | Year | Venue | Task | Architecture | Datasets | GitHub | License | Last update | Reproducibility status | Representation layer | Difficulty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Babaev et al., CoLES ("Contrastive Learning for Event Sequences with Self-Supervision") | 2022 | ACM SIGMOD | Self-supervised sequence embedding (event type + timestamp) | Transformer/RNN encoder + contrastive loss, via `pytorch-lifestream` | Financial transaction sequences (not BPM logs natively) | [dllllb/pytorch-lifestream](https://github.com/dllllb/pytorch-lifestream) | Not confirmed — check repo directly | Active, pip-installable, maintained demos | High confidence: actively maintained library | Pooled encoder output before contrastive projection head, natively exposed | **M — needs a process-mining data loader; core encoder is generic and reusable** |

**Decision: deferred, not adopted.** Not needed by any RQ currently in scope; revisit only if a specific gap emerges.

---

## E. ICPM 2027 / ML4PM workshop CFP

ICPM 2027 is confirmed for **Feb 8–12, 2027, University of Calabria** ([icpmconference.org/2027](https://icpmconference.org/2027/)). No ML4PM-specific workshop page, deadline, or page-limit was retrievable — **no concrete CFP found yet**, most likely because it is too early for the workshop program to be published. Re-check before Phase 11.

---

## Final model roster (finalized 2026-08-14, after systematic-review round)

**9 configurations total** — well beyond the spec's stated 3–6 target, by explicit user direction: the goal shifted from "minimal defensible roster" to "3–4 architecturally distinct models per category to actually run experiments on, decide paper inclusion later." Every model below has either confirmed peer review or (for RLHGNN) a public arXiv preprint with working code, per the relaxed bar the user set for this round.

### Experiment Family A — ecological/SOTA comparison (published models, as-is)

| # | Model | Objective | Architecture | Peer review | Repo | License |
|---|---|---|---|---|---|---|
| A1 | ProcessTransformer | next-event | Transformer (TF/Keras) | arXiv only (disclosed) | `Zaharah/processtransformer` | Apache-2.0 |
| A2 | Camargo et al., GenerativeLSTM | next-event | Stacked LSTM (TF/Keras) | BPM 2019 | `AdaptiveBProcess/GenerativeLSTM` | Apache-2.0 |
| A3 | RLHGNN | next-event | Heterogeneous GNN + RL (PyTorch+DGL) | arXiv 2507.02690, not peer-reviewed (disclosed) | `Joker3993/RLHGNN` | **None — disclosed risk** |
| A4 | SuTraN | full-suffix (+ runtime) | Transformer encoder-decoder (PyTorch) | ICPM 2024 | `BrechtWts/SuffixTransformerNetwork` | MIT |
| A5 | CRTP-LSTM (Gunnarsson et al.) | full-suffix (+ runtime) | Direct/non-autoregressive LSTM (PyTorch) | IEEE TSC 16(4), 2023 | same repo as A4 | MIT |
| A6 | LUPIN | full-suffix | BERT/LLM fine-tuning (PyTorch+HF transformers) | ICPM 2024 | `vinspdb/LUPIN` | CC BY-NC-SA 4.0 (isolate code) |
| A7 | MLMME (Taymouri & La Rosa) | full-suffix (+ runtime) | Adversarial (GAN) RNN encoder-decoder (PyTorch) | SDM 2021 | `farbodtaymouri/MLMME` | GPL-3.0 (isolate code) |

Next-event spans 3 genuinely distinct architectures (Transformer / LSTM / heterogeneous GNN); full-suffix spans 4 (autoregressive Transformer / direct LSTM / BERT-LLM / adversarial RNN). Two license caveats (A6 non-commercial-share-alike, A7 copyleft) and one license gap (A3, no license at all, disclosed) — none of the four full-suffix models' licenses block academic research use, but A6/A7's code should be kept isolated/clearly attributed rather than merged into the project's own permissively-licensed modules, and A3 should be flagged explicitly in the paper.

### Experiment Family B — controlled comparison (one encoder, objective varied)

| # | Model | Objective | Implementation |
|---|---|---|---|
| B1 | Controlled Transformer | next-event | Built in-house (Phase 3), matched architecture/dim/training budget across B1–B2 |
| B2 | Controlled Transformer | full-suffix | Built in-house |

No outcome objective (dropped entirely, see Section C).

### Framework/environment implications (for Phase 2's `pyproject.toml`)

- **TensorFlow** (`tf` optional-dependency group): A1 (ProcessTransformer), A2 (GenerativeLSTM). Both TF/Keras — simpler than expected, only one framework needed for the entire next-event-minus-GNN slice.
- **PyTorch core** (`torch` optional-dependency group): A4 (SuTraN), A5 (CRTP-LSTM), B1, B2.
- **PyTorch + DGL**: A3 (RLHGNN) — needs its own sub-group since DGL has its own version-matching constraints against torch/CUDA.
- **PyTorch + HuggingFace `transformers`**: A6 (LUPIN) — needs its own sub-group (adds a BERT checkpoint download dependency).
- **PyTorch + GPL-3.0 code**: A7 (MLMME) — no extra packages beyond core `torch`, but the licensing isolation note above applies to how its code is vendored/organized in `src/models/`, not to the dependency group.

### Not adopted, explicitly logged as deferred

**I3SP** (independent-lab suffix model, fallback if the "SuTraN/CRTP-LSTM same lab" critique is raised in review) · **TGN-AST** (rejected after hands-on inspection — doesn't deliver GNN diversity, heavy integration cost) · **Wang & Damiani's Time-Aware GNN** (has code, not pursued once RLHGNN was adopted) · **SuTraN+** (outcome-capable, moot since outcome is dropped) · **CoLES/`pytorch-lifestream`** (self-supervised baseline) · **Tax et al.'s original LSTM repo** (superseded by GenerativeLSTM) · every no-repo paper listed in tables A.3/B.2 as a documented "code doesn't exist yet" gap for a possible journal-extension follow-up.

## Compute-estimate status

No candidate paper/repo reported concrete GPU-hour or training-time figures. **Phase 2/3 should run one short empirical timing pilot** (a few epochs on the smallest dataset, across representative models from each framework group) to produce a real compute estimate before committing to running the full 9-model matrix across 5 datasets — that's up to 45 model/dataset combinations, materially more than the original 6-model plan, so this pilot matters more now than it did before.
