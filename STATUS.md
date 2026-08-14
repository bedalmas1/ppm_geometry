# Project Status

> Living tracker. Update this file at the end of every session — new sessions should read this file first, then jump to the relevant phase in [`PLAN.md`](./PLAN.md). Keep entries dated; never delete history from the decision log, only append.

**Current phase:** Phase 2 — Dataset pipeline (partially started)
**Last updated:** 2026-08-14

---

## Phase checklist (mirrors PLAN.md — tick here as work completes)

- [x] Phase 0 — Repository & tooling scaffolding
- [x] Phase 1 — Literature & repository audit
- [ ] Phase 2 — Dataset pipeline
- [ ] Phase 3 — Baseline reproduction
- [ ] Phase 4 — Representation extraction
- [ ] Phase 5 — Geometry metric implementation & synthetic validation
- [ ] Phase 6 — Descriptive geometry study (pilot → full scale)
- [ ] Phase 7 — Reliability study
- [ ] Phase 8 — Future-organization study
- [ ] Phase 9 — Explainability / event-importance study
- [ ] Phase 10 — Controls, ablations & statistical synthesis
- [ ] Phase 11 — Paper

---

## Current phase detail

### Phase 0 — Repository & tooling scaffolding — DONE
- Dependency manager: `uv` (installed via `pip install --user uv`; not on this machine's PATH inside the bash tool, so `python -m uv <cmd>` was used — works identically; a normal shell with `uv`'s install location on PATH will have `uv` directly).
- `pyproject.toml`: project metadata, `requires-python = ">=3.11,<3.14"` (permissive placeholder — Phase 1 may need to narrow this if a chosen baseline repo requires an older torch/Python combination), `[tool.uv] package = false` (this is a research repo, not a distributable package — imports rely on `pythonpath = ["src"]` in pytest config, no editable install needed), `dev` dependency group with `pytest`.
- `uv.lock` generated and committed.
- `.venv/` created locally (gitignored).
- `.gitignore` added: excludes `.venv/`, caches, `data/raw/*` and `data/processed/*` (keeps `.gitkeep`), `results/*` and `figures/*` (keeps `.gitkeep`), checkpoint file extensions, notebook checkpoints, editor/OS cruft. Explicitly keeps `uv.lock`, `PLAN.md`, `STATUS.md`.
- `tests/test_scaffold.py` placeholder + `pytest` config (`testpaths`, `pythonpath`) — `uv run pytest -q` passes (1 test). Removed `tests/.gitkeep` since a real file now lives there.
- `configs/README.md` documents the config convention (one YAML per experiment; `datasets/`, `models/`, `experiments/` subdirectories to be created as needed) — replaced `configs/.gitkeep`.
- `README.md` written: project summary, setup instructions, full repo layout, pointers to STATUS/PLAN/CLAUDE.

Not yet decided in Phase 0 (deferred, not blocking): experiment tracking/logging format (lightweight structured logs vs. external tracker) — will decide once Phase 3 (baseline reproduction) actually needs to log training runs.

### Phase 1 — Literature & repository audit — DONE (including user-directed revision)
Full write-up: [`paper/related_work_model_audit.md`](./paper/related_work_model_audit.md).

**Original pass (6 configs) completed 2026-08-14**, then the user reviewed it and pushed back: a single model per Family A category (next-event, full-suffix) is too thin for ICPM/ML4PM reviewers, especially since ProcessTransformer's own venue provenance is arXiv-only and CRTP-LSTM's original paper's peer-review venue was initially unconfirmed. Final decisions (same day, all resolved):
1. Drop outcome prediction **entirely** (not just from Family A — the planned Family B outcome slot, B3, is also dropped). H3 is now explicitly untested/deferred, stated plainly rather than hidden.
2. Next-event Family A gets a second model: **Camargo et al.'s GenerativeLSTM** (BPM 2019, peer-reviewed, Apache-2.0, `AdaptiveBProcess/GenerativeLSTM`, actively maintained through Oct 2025) — a genuine Transformer-vs-LSTM architectural contrast within the same objective.
3. Full-suffix Family A's second model: a targeted follow-up search resolved CRTP-LSTM's provenance rather than replacing it — **it is Gunnarsson, vanden Broucke & De Weerdt, IEEE Transactions on Services Computing 16(4), 2023**, a genuinely peer-reviewed journal paper. The search also surfaced **I3SP** (BPM 2025 Workshops/LNBIP 569, independent KU-Leuven-unaffiliated authors, more recent) as an option to defuse a subtler "SuTraN+CRTP-LSTM are the same lab/codebase" critique — user opted to keep the roster at SuTraN+CRTP-LSTM only (both now confirmed independently peer-reviewed, which was the actual bar), logging I3SP as a deferred candidate rather than adopting it.

**Final roster shape:** 2 next-event + 2 full-suffix (Family A) + 1 next-event + 1 full-suffix (Family B, controlled comparison) = **6 configs**, no outcome objective.

- Family A (ecological/SOTA, published models as-is): **A1** ProcessTransformer (next-event, TF/Keras, Apache-2.0, `Zaharah/processtransformer`, arXiv-only provenance disclosed as a known weakness), **A2** Camargo GenerativeLSTM (next-event, Apache-2.0, `AdaptiveBProcess/GenerativeLSTM`, BPM 2019), **A3** SuTraN (full-suffix + remaining runtime, PyTorch, MIT, `BrechtWts/SuffixTransformerNetwork`, ICPM 2024), **A4** CRTP-LSTM (full-suffix + remaining runtime, LSTM, same repo/license as A3, IEEE TSC 2023).
- Family B (controlled comparison, one encoder architecture, objective varied, built in-house in Phase 3): **B1** Transformer-next, **B2** Transformer-suffix. No B3/outcome.

**Rejected/deferred candidates (see the audit doc for full reasoning):** Tax et al.'s original LSTM repo (stale, unlicensed, pre-TF2 — superseded by the now-adopted Camargo `GenerativeLSTM`); no dedicated Family A outcome-prediction model and no Family B outcome model either (classical-ML-dominated field, no usable continuous z_t, and the objective is being set aside for now rather than filled with a weak baseline); **I3SP** (BPM 2025/LNBIP 569, independent-lab suffix model — deliberately not adopted; logged as the fallback if a reviewer specifically raises the "SuTraN/CRTP-LSTM are the same lab" concern); SuTraN+ (outcome-capable SuTraN extension — moot now that outcome is dropped, kept on record); CoLES/`pytorch-lifestream` self-supervised baseline (strong library, but adapting it from financial-transaction to event-log data is nontrivial and not needed by any RQ yet — deferred); ED-LSTM/SEP-LSTM (also in the SuTraN repo, available if a broader LSTM sweep is ever wanted, not part of the initial roster); SPICE reimplementation (useful correctness reference for known leakage/BatchNorm/time-scaling bugs, but CC BY-NC-ND license blocks using it as a code source).

**Hands-on verification (2026-08-14):** ProcessTransformer and SuTraN repos were both `git clone`d to a scratch location (not committed) and their source read directly:
- ProcessTransformer: confirmed Apache-2.0 license text, confirmed single-`TransformerBlock` → `GlobalAveragePooling1D` architecture. z_t = the pooled output (before the `Dense(64)`/output head) — the `Dense(64)` hidden layer is a ready-made second option for the Phase 10 "representation layer" ablation. **Finding:** `setup.py` has no version pins beyond `tensorflow>=2.4` — needs pinning in Phase 3 per this project's reproducibility conventions.
- SuTraN: confirmed MIT license text, confirmed **PyTorch** (not TF — useful framework contrast with ProcessTransformer). z_t = the encoder-stack output tensor (`batch × window_size × d_model`), used as the decoder's cross-attention memory before any prediction head — an exact per-prefix-position match for the spec's z_t trajectory definition, extracted in one forward pass. No top-level `requirements.txt`; dependencies must be inferred from imports and pinned ourselves.
- Camargo's `GenerativeLSTM` has **not yet** been hands-on verified (cloned/source-read) — this is an outstanding Phase 2/3 task, same treatment as A1/A3 got.

**Not resolved by literature search (as expected — flagged in the audit doc rather than guessed):** concrete GPU-hour/training-time figures for any candidate. Recommendation: Phase 2/3 should run a short empirical timing pilot (a few epochs, smallest dataset, A1+B1) rather than estimate from papers that don't report it.

**ICPM 2027 / ML4PM CFP:** conference dates confirmed (Feb 8–12, 2027, University of Calabria); no ML4PM-specific CFP, deadline, or page limit found yet — too early. Re-check before Phase 11.

### Phase 2 — Dataset pipeline — partially started
- **Dataset roster chosen (pending final write-up):** BPIC12, BPIC17, BPIC19, Sepsis, Helpdesk — 5 logs spanning ~1K–250K cases, 16–42 activities, simple/linear to highly variable/rework-heavy structure. Verified 2026-08-14 via a research fork: all 5 have live, confirmed 4TU.ResearchData download links (not guessed) — BPIC12/17/19 and Sepsis are XES(.gz), **Helpdesk is CSV** (needs a separate ingestion path from the shared preprocessing pipeline). BPIC17/BPIC19 overlap with the SuTraN/CRTP-LSTM repo's own datasets, which helps the Phase 3 reproduction check. Sanity check: roster judged not cherry-picked; only minor flag is Helpdesk having the weakest native outcome label of the five, which doesn't matter now that outcome prediction is dropped.
- **Environment/dependency setup started:** `pyproject.toml` now has base deps (`pandas`, `numpy`, `scikit-learn`, `pyyaml`) synced and locked, plus two optional-dependency groups added (not yet synced/installed): `tf` (`tensorflow>=2.16,<2.20`, for A1 ProcessTransformer only) and `torch` (`torch>=2.3`, for A3/A4/B1/B2), kept separate so a training run for one framework never resolves the other's wheels. This answers the Phase 1 "TF vs PyTorch coexistence" open question: **separate optional-dependency groups within one project**, not fully isolated venvs or per-model repos.
- **Not yet done:** actually downloading the 5 datasets, writing the shared preprocessing/splitting pipeline, computing the spec §6 descriptive-stats table, the empirical compute-timing pilot, and finalizing the dataset list into a committed doc (currently only in this STATUS.md entry, not yet in `paper/` or `configs/datasets/`).

---

## Decision log

Append one entry per decision, dated, with a one-line rationale. Never edit past entries — supersede them with a new dated entry if a decision changes.

- **2026-08-14** — Created PLAN.md/STATUS.md/CLAUDE.md tracking system and empty reproducibility directory skeleton per spec §21. No implementation work started yet.
- **2026-08-14** — Chose `uv` as the Python dependency manager (user confirmed over pip+pip-tools and poetry). Installed via `pip install --user uv` since it wasn't preinstalled.
- **2026-08-14** — Completed Phase 0: pyproject.toml (`requires-python>=3.11,<3.14`, non-package project via `[tool.uv] package = false`), uv.lock, .gitignore, pytest wiring with a passing placeholder test, configs/README.md, and README.md. Deferred the experiment-tracking-format decision to Phase 3 (not needed until real training runs exist).
- **2026-08-14** — Completed Phase 1 literature/repository audit via three parallel research passes + hands-on clone verification of the two Family A picks. Finalized an initial 6-model roster (A1 ProcessTransformer, A2 SuTraN, A3 CRTP-LSTM, B1/B2/B3 controlled Transformer variants) — see `paper/related_work_model_audit.md` for full reasoning and rejected alternatives.
- **2026-08-14** — User reviewed the roster and judged one model per Family A category too weak for reviewers. Decided: (a) drop outcome prediction entirely, not just from Family A — no Family B outcome model either, H3 explicitly deferred/untested; (b) promote Camargo et al.'s `GenerativeLSTM` (BPM 2019, peer-reviewed, actively maintained) to a full second Family A next-event model, giving a Transformer-vs-LSTM architectural contrast; (c) run one more targeted research pass for a second full-suffix Family A model with independently-confirmed peer review, rather than default to CRTP-LSTM (whose original paper's venue was never confirmed) — user explicitly requires this choice to be thoroughly justified in the paper itself, not just logged internally.
- **2026-08-14** — Follow-up search resolved CRTP-LSTM's provenance: it is Gunnarsson et al., IEEE Transactions on Services Computing 16(4), 2023 — genuinely peer-reviewed. Also surfaced I3SP (BPM 2025 Workshops/LNBIP 569, independent authors) as an option to address a "SuTraN/CRTP-LSTM are the same lab" critique. User decided to finalize the roster at SuTraN+CRTP-LSTM (both now confirmed peer-reviewed, independence bar met) rather than add I3SP as a third suffix model — keeps the roster at 6 total, I3SP logged as a deferred fallback. **Phase 1 is now fully closed** with a final 6-model roster: A1 ProcessTransformer, A2 Camargo GenerativeLSTM, A3 SuTraN, A4 CRTP-LSTM, B1/B2 controlled Transformer (next-event/suffix), no outcome objective.
- **2026-08-14** — Chose the TF/PyTorch coexistence strategy for Phase 2/3: two separate `pyproject.toml` optional-dependency groups (`tf`, `torch`) in one project, rather than fully isolated per-model environments — keeps one lockfile/one repo while still letting each framework be installed independently.
- **2026-08-14** — Chose the 5-dataset roster (BPIC12, BPIC17, BPIC19, Sepsis, Helpdesk) for Phase 2, verified via a research fork that all 5 have live 4TU.ResearchData download links and that the roster isn't cherry-picked for structural diversity. Not yet written up as a committed dataset-selection doc.

---

## Open questions / not yet decided

- Experiment tracking/logging format (lightweight structured logs under `results/` vs. an external tracker) — deferred until Phase 3 needs to log actual training runs.
- Number of seeds per model/dataset — depends on compute budget; no literature source reported concrete GPU-hour figures, so this needs an empirical timing pilot in Phase 2/3 rather than a literature-based estimate.
- `requires-python` in `pyproject.toml` (`>=3.11,<3.14`) is now paired with two optional-dependency groups (`tf`, `torch`) rather than a single pin — still needs exact-version pinning once Phase 3 settles a CUDA/CPU target (ProcessTransformer's `setup.py` itself is unpinned beyond `tensorflow>=2.4`; SuTraN/CRTP-LSTM have no dependency file at all).
- ICPM 2027 ML4PM workshop CFP details (page limit, format, deadline) — checked 2026-08-14, not published yet; conference dates (Feb 8–12, 2027) are confirmed. Re-check closer to Phase 11.
- Dataset roster (BPIC12/17/19, Sepsis, Helpdesk) is chosen and link-verified but not yet written up as a committed doc (e.g. `configs/datasets/` or a `paper/` section) — do that as part of closing Phase 2's early tasks.

## Blockers

None currently.

## Next steps (pick these up first in the next session)

1. Continue Phase 2 — dataset pipeline: write up the chosen 5-dataset roster (BPIC12, BPIC17, BPIC19, Sepsis, Helpdesk) as a committed doc, then implement shared preprocessing/splitting and compute the spec §6 descriptive-stats table.
2. Sync and smoke-test the new `tf`/`torch` optional-dependency groups (e.g. `uv sync --extra torch` and `--extra tf` separately) to confirm they actually resolve cleanly before Phase 3 needs them.
3. Hands-on verify Camargo's `GenerativeLSTM` and pin down CRTP-LSTM's exact z_t extraction point (both still outstanding — ProcessTransformer and SuTraN already got this treatment in Phase 1).
4. Once a dataset exists, run the short empirical timing pilot noted above (a few epochs, smallest dataset, A1+B1) to get a real compute estimate before scaling to the full matrix.
