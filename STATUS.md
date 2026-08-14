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

### Phase 1 — Literature & repository audit — DONE (three rounds: initial → user-directed strengthening → systematic review)
Full write-up: [`paper/related_work_model_audit.md`](./paper/related_work_model_audit.md) — this is now the authoritative source for all model-selection reasoning; this section is a summary only.

**Round 1** (initial ad-hoc pass): 6-model roster (2 Family A + 1+1 Family B, no outcome). **Round 2** (user pushback: single model per Family A category too thin for reviewers): added Camargo's GenerativeLSTM as 2nd next-event model, resolved CRTP-LSTM's peer-review provenance (IEEE TSC 2023) as 2nd suffix model, dropped outcome prediction entirely (not just from Family A). **Round 3** (user flagged a real gap — GNN-based next-event models had never been searched for — and asked for a *systematic* review rather than sequential ad-hoc passes, with a documented methodology): ran dedicated systematic searches (2021–2026) for next-event, GNN-specifically, and full-suffix, then hands-on cloned/verified every new candidate. User explicitly relaxed the adoption bar for this round to "peer-reviewed or arXiv preprint + usable repo," and redefined the goal as "3–4 architecturally distinct models per category to experiment with, decide paper inclusion later."

**Key finding from the systematic review:** next-event and full-suffix are *not symmetric* in the literature — next-event PPM research (2021–2026) is high-publication/low-code (of ~17 papers found across both the GNN-specific and general passes, only 2 had any repo, and one of those turned out not to deliver what it claimed on hands-on inspection), while full-suffix is comparatively rich in reproducible work (4 models cleared the bar: SuTraN, CRTP-LSTM, LUPIN, MLMME, spanning 4 distinct architectural paradigms).

**Final roster: 9 configurations** (well beyond the spec's stated 3–6, by explicit user direction):

- Family A next-event (3): **A1** ProcessTransformer (Transformer, TF/Keras, Apache-2.0, arXiv-only provenance disclosed) · **A2** Camargo GenerativeLSTM (stacked LSTM, TF/Keras, Apache-2.0, BPM 2019) · **A3** RLHGNN (heterogeneous GNN + RL, PyTorch+DGL, arXiv 2507.02690 not peer-reviewed, **no LICENSE file at all — disclosed risk**, the only candidate found across two searches that delivers genuine GNN architectural diversity with working single-case code).
- Family A full-suffix (4): **A4** SuTraN (Transformer enc-dec, PyTorch, MIT, ICPM 2024) · **A5** CRTP-LSTM (direct/non-autoregressive LSTM, PyTorch, MIT, IEEE TSC 2023) · **A6** LUPIN (BERT/LLM fine-tuning, PyTorch+HF transformers, **CC BY-NC-SA 4.0** — isolate its code, don't merge into permissively-licensed modules, ICPM 2024, fully version-pinned `requirements.txt` — best reproducibility hygiene of any model audited) · **A7** MLMME (Taymouri & La Rosa, adversarial/GAN-trained RNN enc-dec, PyTorch, **GPL-3.0** — isolate its code, SDM 2021).
- Family B (2, unchanged): **B1** Transformer-next, **B2** Transformer-suffix, built in-house Phase 3. No outcome objective.

**A notable hands-on correction**: TGN-AST (Hennig & Schmidt, BPM 2025 main track — stronger peer review than ProcessTransformer, MIT-licensed) was initially considered as the GNN-diversity pick, but reading its actual source revealed it is a **TensorFlow/Keras Transformer variant**, not an end-to-end GNN — the graph component is a decoupled pretraining step producing an input feature. Rejected: doesn't deliver the architectural diversity it was sought for, and is Colab-oriented/notebook-only with a 642MB repo. RLHGNN took its place instead.

**Licensing summary across the full roster:** Apache-2.0 (A1, A2) · MIT (A4, A5) · CC BY-NC-SA 4.0, isolate (A6) · GPL-3.0, isolate (A7) · **no license at all, disclosed risk** (A3). No license blocks academic research use, but A6/A7's code must stay isolated/clearly attributed (same handling as SPICE's ND license got), and A3's gap must be stated explicitly in the paper — consider emailing RLHGNN's authors for clarification.

**Hands-on clone verification completed for:** ProcessTransformer, SuTraN, CRTP-LSTM (architecture confirmed, exact z_t layer still TBD), Camargo GenerativeLSTM, LUPIN, MLMME, RLHGNN, TGN-AST (rejected after inspection). All scratch-only clones, nothing added to this repo.

**New findings requiring Phase 3 follow-up:** ProcessTransformer's `setup.py` unpinned beyond `tensorflow>=2.4`; Camargo's `environment.yml` pulls an unpinned git dependency over **plain HTTP** (`git+http://github.com/Mcamargo85/support_modules.git`) — fix in Phase 3; SuTraN/CRTP-LSTM/RLHGNN have no dependency files at all, need pinning ourselves.

**Framework groups needed (feeds Phase 2's `pyproject.toml`, already added):** `tf` (A1, A2) · `torch` (A4, A5, A7, B1, B2) · `torch-hf` (A6, adds HuggingFace `transformers`) · `torch-dgl` (A3, adds DGL, isolated since DGL has its own torch/CUDA version-matching constraints).

**Rejected/deferred, logged not silently dropped:** I3SP (independent-lab suffix model, fallback if a "SuTraN/CRTP-LSTM same lab" critique arises), TGN-AST, Wang & Damiani's Time-Aware GNN (has code, not pursued once RLHGNN was adopted), SuTraN+, CoLES/`pytorch-lifestream`, Tax et al.'s original LSTM repo, every no-repo paper found in the systematic searches (documented as a "code doesn't exist yet" gap, candidate for a journal-extension follow-up — chase authors or reimplement), no outcome-prediction model (dropped entirely, H3 explicitly untested).

**ICPM 2027 / ML4PM CFP:** conference dates confirmed (Feb 8–12, 2027, University of Calabria); no ML4PM-specific CFP, deadline, or page limit found yet — too early. Re-check before Phase 11.

### Phase 2 — Dataset pipeline — partially started
- **Dataset roster chosen (pending final write-up):** BPIC12, BPIC17, BPIC19, Sepsis, Helpdesk — 5 logs spanning ~1K–250K cases, 16–42 activities, simple/linear to highly variable/rework-heavy structure. Verified 2026-08-14 via a research fork: all 5 have live, confirmed 4TU.ResearchData download links (not guessed) — BPIC12/17/19 and Sepsis are XES(.gz), **Helpdesk is CSV** (needs a separate ingestion path from the shared preprocessing pipeline). BPIC17/BPIC19 overlap with the SuTraN/CRTP-LSTM repo's own datasets, which helps the Phase 3 reproduction check. Sanity check: roster judged not cherry-picked; only minor flag is Helpdesk having the weakest native outcome label of the five, which doesn't matter now that outcome prediction is dropped.
- **Environment/dependency setup, revised for the 9-model roster:** `pyproject.toml` has base deps (`pandas`, `numpy`, `scikit-learn`, `pyyaml`) synced and locked, plus four optional-dependency groups (resolved/locked via `uv lock`, not yet synced/installed into the active venv): `tf` (A1, A2), `torch` (A4, A5, A7, B1, B2), `torch-hf` (A6, adds `transformers`), `torch-dgl` (A3, adds `dgl`, isolated since DGL has its own torch/CUDA version constraints). This answers the Phase 1 "TF vs PyTorch coexistence" question and its extension to two more frameworks: **separate optional-dependency groups within one project**, not fully isolated venvs or per-model repos.
- **Not yet done:** actually downloading the 5 datasets, writing the shared preprocessing/splitting pipeline, computing the spec §6 descriptive-stats table, the empirical compute-timing pilot (now more important given 9 models × 5 datasets = up to 45 combinations), smoke-testing the 4 dependency groups actually install cleanly, and finalizing the dataset list into a committed doc (currently only in this STATUS.md entry, not yet in `paper/` or `configs/datasets/`).

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
- **2026-08-14** — User flagged that GNN-based next-event models were never searched for in Phase 1, and questioned whether the ML4PM paper should narrow to next-event-only (broad architecture sweep) with full-suffix+outcome deferred to a journal extension. Rather than decide on assumptions, ran a GNN-specific audit first: found ~10 GNN-for-PPM papers but only 2 with any repo, both unpeer-reviewed 2025 preprints — confirmed the category is too thin to justify re-scoping, and the user kept the paper's cross-objective design.
- **2026-08-14** — User then asked for a full systematic literature review (not sequential ad-hoc audits) to avoid missing other categories. Scoped to next-event + full-suffix only (outcome stays dropped), 2021–2026. Ran two more systematic passes; user explicitly relaxed the adoption bar to "peer-reviewed or arXiv preprint + usable repo" and redefined the goal as 3–4 architecturally distinct models per category for experimentation, paper-inclusion decided later.
- **2026-08-14** — Systematic full-suffix pass found LUPIN (ICPM 2024, BERT-based) and MLMME (SDM 2021, GAN-based) as 2 more reproducible candidates. User decided: adopt both, bringing full-suffix to 4 models (SuTraN, CRTP-LSTM, LUPIN, MLMME) spanning 4 architectural paradigms.
- **2026-08-14** — Systematic next-event pass found TGN-AST (BPM 2025 main track) as a candidate; hands-on source inspection revealed it's actually a TF/Keras Transformer variant (graph component is a decoupled pretraining step, not an end-to-end GNN) — rejected despite its strong peer-review venue. User decided to adopt RLHGNN instead (the one GNN audit candidate with real, working single-case code) despite RLHGNN having no LICENSE file at all — a disclosed risk, not a hidden one. Final next-event roster: 3 models (ProcessTransformer, GenerativeLSTM, RLHGNN).
- **2026-08-14** — **Phase 1 is now fully and finally closed** at a 9-configuration roster (A1–A7 Family A + B1/B2 Family B), well beyond the spec's stated 3–6 by explicit user direction. `pyproject.toml` extended with `torch-hf` (transformers, for LUPIN) and `torch-dgl` (dgl, for RLHGNN) optional-dependency groups alongside the existing `tf`/`torch`; `uv lock` re-run and confirmed resolving cleanly.

---

## Open questions / not yet decided

- Experiment tracking/logging format (lightweight structured logs under `results/` vs. an external tracker) — deferred until Phase 3 needs to log actual training runs.
- Number of seeds per model/dataset — depends on compute budget; no literature source reported concrete GPU-hour figures, so this needs an empirical timing pilot in Phase 2/3 rather than a literature-based estimate. More important now given 9 models × 5 datasets = up to 45 combinations.
- `requires-python` in `pyproject.toml` (`>=3.11,<3.14`) now spans 4 optional-dependency groups (`tf`, `torch`, `torch-hf`, `torch-dgl`) — still needs exact-version pinning once Phase 3 settles a CUDA/CPU target. Several models have no pinned/no dependency file at all: ProcessTransformer (`tensorflow>=2.4` only), Camargo GenerativeLSTM (pulls an unpinned git dependency over plain HTTP — needs fixing, not just pinning), SuTraN/CRTP-LSTM/RLHGNN (no dependency file).
- ICPM 2027 ML4PM workshop CFP details (page limit, format, deadline) — checked 2026-08-14, not published yet; conference dates (Feb 8–12, 2027) are confirmed. Re-check closer to Phase 11.
- Dataset roster (BPIC12/17/19, Sepsis, Helpdesk) is chosen and link-verified but not yet written up as a committed doc (e.g. `configs/datasets/` or a `paper/` section) — do that as part of closing Phase 2's early tasks.
- **RLHGNN's missing license** — worth deciding whether to email the authors for clarification before relying on it further, and exactly how to disclose the gap in the paper.
- **Which of the 9 models actually make it into the ML4PM workshop paper vs. stay as "ran the experiment, held for a journal extension"** — user explicitly deferred this decision ("we'll see later if we integrate them in the paper"). Worth revisiting once Phase 6's descriptive results exist and it's clearer which comparisons are most informative.
- Whether to pursue any of the ~15 no-repo papers found during the systematic review (emailing authors, reimplementing) — logged as a journal-extension-scale task, not blocking the workshop paper.

## Blockers

None currently.

## Next steps (pick these up first in the next session)

1. Continue Phase 2 — dataset pipeline: write up the chosen 5-dataset roster (BPIC12, BPIC17, BPIC19, Sepsis, Helpdesk) as a committed doc, then implement shared preprocessing/splitting and compute the spec §6 descriptive-stats table.
2. Sync and smoke-test all 4 dependency groups (`uv sync --extra tf`, `--extra torch`, `--extra torch-hf`, `--extra torch-dgl`, tried separately) to confirm they actually resolve/install cleanly before Phase 3 needs them — `torch-dgl` in particular is a risk given DGL's version-matching finickiness.
3. Hands-on-pin CRTP-LSTM's exact z_t extraction layer (architecture confirmed, exact layer still TBD — every other adopted model already has this pinned down).
4. Fix Camargo GenerativeLSTM's unpinned-HTTP git dependency (`git+http://github.com/Mcamargo85/support_modules.git`) before relying on it for training.
5. Once a dataset exists, run the short empirical timing pilot (a few epochs, smallest dataset, one model per framework group) to get a real compute estimate before scaling to the full 9-model × 5-dataset matrix.
