# Project Status

> Living tracker. Update this file at the end of every session — new sessions should read this file first, then jump to the relevant phase in [`PLAN.md`](./PLAN.md). Keep entries dated; never delete history from the decision log, only append.

**Current phase:** Phase 2 — Dataset pipeline (not started)
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

### Phase 1 — Literature & repository audit — DONE
Full write-up: [`paper/related_work_model_audit.md`](./paper/related_work_model_audit.md).

**Final model roster (6 configurations):**
- Family A (ecological/SOTA, published models as-is): **A1** ProcessTransformer (next-event, TF/Keras, Apache-2.0, `Zaharah/processtransformer`), **A2** SuTraN (full-suffix + remaining runtime, PyTorch, MIT, `BrechtWts/SuffixTransformerNetwork`), **A3** CRTP-LSTM (full-suffix, LSTM, same repo/license as A2 — adopted as the optional LSTM baseline at near-zero extra integration cost).
- Family B (controlled comparison, one encoder architecture, objective varied, built in-house in Phase 3): **B1** Transformer-next, **B2** Transformer-suffix, **B3** Transformer-outcome.

**Rejected/deferred candidates (see the audit doc for full reasoning):** Tax et al.'s original LSTM repo (stale, unlicensed, pre-TF2 — superseded by Camargo's `GenerativeLSTM`, which itself wasn't needed once CRTP-LSTM was available for free in the SuTraN repo); no dedicated Family A outcome-prediction model (classical-ML-dominated field, no usable continuous z_t — outcome supervision comes entirely from B3); SuTraN+ (outcome-capable SuTraN extension, unconfirmed peer-review status — watch item); CoLES/`pytorch-lifestream` self-supervised baseline (strong library, but adapting it from financial-transaction to event-log data is nontrivial and not needed by any RQ yet — deferred); ED-LSTM/SEP-LSTM (also in the SuTraN repo, available if a broader LSTM sweep is ever wanted, not part of the initial roster); SPICE reimplementation (useful correctness reference for known leakage/BatchNorm/time-scaling bugs, but CC BY-NC-ND license blocks using it as a code source).

**Hands-on verification (2026-08-14):** ProcessTransformer and SuTraN repos were both `git clone`d to a scratch location (not committed) and their source read directly:
- ProcessTransformer: confirmed Apache-2.0 license text, confirmed single-`TransformerBlock` → `GlobalAveragePooling1D` architecture. z_t = the pooled output (before the `Dense(64)`/output head) — the `Dense(64)` hidden layer is a ready-made second option for the Phase 10 "representation layer" ablation. **Finding:** `setup.py` has no version pins beyond `tensorflow>=2.4` — needs pinning in Phase 3 per this project's reproducibility conventions.
- SuTraN: confirmed MIT license text, confirmed **PyTorch** (not TF — useful framework contrast with ProcessTransformer). z_t = the encoder-stack output tensor (`batch × window_size × d_model`), used as the decoder's cross-attention memory before any prediction head — an exact per-prefix-position match for the spec's z_t trajectory definition, extracted in one forward pass. No top-level `requirements.txt`; dependencies must be inferred from imports and pinned ourselves.

**Not resolved by literature search (as expected — flagged in the audit doc rather than guessed):** concrete GPU-hour/training-time figures for any candidate. Recommendation: Phase 2/3 should run a short empirical timing pilot (a few epochs, smallest dataset, A1+B1) rather than estimate from papers that don't report it.

**ICPM 2027 / ML4PM CFP:** conference dates confirmed (Feb 8–12, 2027, University of Calabria); no ML4PM-specific CFP, deadline, or page limit found yet — too early. Re-check before Phase 11.

### Phase 2 — Dataset pipeline
Not started. See PLAN.md Phase 2 for the full task list. Dataset selection (4–6 logs) should draw on BPI Challenge logs per spec §6, still to be finalized.

---

## Decision log

Append one entry per decision, dated, with a one-line rationale. Never edit past entries — supersede them with a new dated entry if a decision changes.

- **2026-08-14** — Created PLAN.md/STATUS.md/CLAUDE.md tracking system and empty reproducibility directory skeleton per spec §21. No implementation work started yet.
- **2026-08-14** — Chose `uv` as the Python dependency manager (user confirmed over pip+pip-tools and poetry). Installed via `pip install --user uv` since it wasn't preinstalled.
- **2026-08-14** — Completed Phase 0: pyproject.toml (`requires-python>=3.11,<3.14`, non-package project via `[tool.uv] package = false`), uv.lock, .gitignore, pytest wiring with a passing placeholder test, configs/README.md, and README.md. Deferred the experiment-tracking-format decision to Phase 3 (not needed until real training runs exist).
- **2026-08-14** — Completed Phase 1 literature/repository audit via three parallel research passes + hands-on clone verification of the two Family A picks. Finalized the 6-model roster (A1 ProcessTransformer, A2 SuTraN, A3 CRTP-LSTM, B1/B2/B3 controlled Transformer variants) — see `paper/related_work_model_audit.md` for full reasoning and rejected alternatives. No dedicated Family A outcome model adopted; outcome supervision comes entirely from the in-house controlled Transformer (B3), since the outcome-prediction literature is classical-ML-dominated with no usable continuous representation.

---

## Open questions / not yet decided

- Final dataset roster (4–6 logs) — pending Phase 2; should include BPIC17/BPIC19 (already used by SuTraN/CRTP-LSTM, convenient for reproduction) plus enough additional logs to cover the structural variation spec §6 requires (loops/rework, outcome diversity, structuredness) without cherry-picking for pretty geometry.
- Experiment tracking/logging format (lightweight structured logs under `results/` vs. an external tracker) — deferred until Phase 3 needs to log actual training runs.
- Number of seeds per model/dataset — depends on compute budget; no literature source reported concrete GPU-hour figures, so this needs an empirical timing pilot in Phase 2/3 rather than a literature-based estimate.
- `requires-python` upper/lower bound in `pyproject.toml` is a placeholder (`>=3.11,<3.14`). Now informed by Phase 1: ProcessTransformer needs TensorFlow (unpinned, `>=2.4`, pre-Keras-3 style — may need a `tf_keras` shim on newer TF), SuTraN/CRTP-LSTM need PyTorch (no top-level dependency file — versions to be pinned ourselves). Both frameworks will need to coexist in the same environment (or be isolated per-model) — decide the approach in Phase 2.
- ICPM 2027 ML4PM workshop CFP details (page limit, format, deadline) — checked 2026-08-14, not published yet; conference dates (Feb 8–12, 2027) are confirmed. Re-check closer to Phase 11.

## Blockers

None currently.

## Next steps (pick these up first in the next session)

1. Begin Phase 2 — dataset pipeline (see PLAN.md Phase 2 tasks): finalize the 4–6 dataset roster (BPIC17/BPIC19 are a natural starting point given the roster's model choices), implement shared preprocessing/splitting, compute the required per-dataset descriptive stats.
2. Decide the TensorFlow-vs-PyTorch environment strategy needed to run A1 (TF) alongside A2/A3/B1–B3 (PyTorch) — e.g. one shared env with both installed and pinned, or per-model isolated envs. Resolve the unpinned-dependency findings from Phase 1 (pin exact versions for both frameworks) at the same time.
3. Once a dataset exists, run the short empirical timing pilot noted above (a few epochs, smallest dataset, A1+B1) to get a real compute estimate before scaling to the full matrix.
