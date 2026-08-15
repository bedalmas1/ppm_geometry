# Phase 2 — Dataset Selection

Produced per [`PLAN.md`](../PLAN.md) Phase 2 / spec §6. Finalizes the dataset roster chosen 2026-08-14 (see `STATUS.md`'s decision log) and records the verification/sanity-check evidence behind it. Per-dataset source/split configuration lives in `configs/datasets/*.yaml`; this document is the narrative rationale plus the descriptive-statistics table spec §6 requires.

## Roster

Computed programmatically by `scripts/prepare_dataset.py` from the actual parsed logs (2026-08-15) — not hand-copied from any paper or landing page, per PLAN.md's cross-cutting reminder. Full per-dataset provenance (git commit, software versions, split hash) lives in `data/processed/<dataset>/manifest.json` (gitignored, regenerate via `scripts/prepare_dataset.py --all`).

| Dataset | Cases | Events | Activities | Mean / median trace length | Trace variants | Variant entropy (bits) | Temporal span | Train / val / test cases |
|---|---|---|---|---|---|---|---|---|
| **BPIC12** | 13,087 | 262,200 | 24 | 20.04 / 11.0 | 4,366 | 7.75 | 2011-09-30 → 2012-03-14 | 8,376 / 2,094 / 2,617 |
| **BPIC17** | 31,509 | 1,202,267 | 26 | 38.16 / 35.0 | 15,930 | 11.99 | 2016-01-01 → 2017-02-01 | 20,166 / 5,041 / 6,302 |
| **BPIC19** | 251,734 | 1,595,923 | 42 | 6.34 / 5.0 | 11,973 | 6.24 | **see note below** | 161,110 / 40,277 / 50,347 |
| **Sepsis** | 1,050 | 15,214 | 16 | 14.49 / 13.0 | 846 | 9.33 | 2013-11-07 → 2015-06-05 | 672 / 168 / 210 |
| **Helpdesk** | 4,580 | 21,348 | 14 | 4.66 / 4.0 | 226 | 3.36 | 2010-01-13 → 2014-01-03 | 2,931 / 733 / 916 |

Every case/event/activity count above matches the figures reported in the originating papers/landing pages exactly (BPIC12: 13,087 cases/262,200 events; BPIC19: 251,734 cases/1,595,923 events/42 activities; etc.) — a useful independent cross-check that the parsing pipeline is correct.

**BPIC19 temporal-span data-quality note:** the raw computed span is 1948-01-26 → 2020-04-09, which is not a real 72-year process — this is a known BPIC19 data-quality artifact (a small number of events carry placeholder/default timestamps far outside the log's actual ~2018 collection period), not a pipeline bug. This needs a documented cleaning rule before Phase 3 training (e.g. drop or clip events outside a sane date range) rather than being silently accepted — logged here as a Phase 2/3 open item, not yet resolved.

Structural roles: BPIC12 (classic, well-studied loops; loan application), BPIC17 (large, branching, loan application; shared with the SuTraN/CRTP-LSTM repo A4/A5), BPIC19 (largest, purchase-order handling, high variant count; also shared with A4/A5), Sepsis (smallest, hospital process, high rework/variability — variant entropy of 9.33 bits across only 1,050 cases confirms this), Helpdesk (small, simple/linear ticketing — lowest variant entropy of the five at 3.36 bits, confirming its "simple/linear" structural role; CSV, not XES; same dataset ProcessTransformer's own paper evaluated on).

## Why these five, not others

Verified 2026-08-14 via a research fork before being committed to: all five have live, confirmed 4TU.ResearchData download links (not guessed — see `configs/datasets/*.yaml` for the exact DOIs/URLs). The fork's independent sanity check on structural diversity: the roster spans ~1K to ~250K cases and 16–42 activities, and simple/linear (Helpdesk) to highly variable/rework-heavy (Sepsis, BPIC19) structure — matching combinations used across dozens of established PPM papers (Tax, Camargo, the Teinemaa outcome-prediction benchmark). One flag raised and accepted: Helpdesk has the weakest native outcome label of the five, which doesn't matter now that outcome prediction is dropped from this study entirely (see `paper/related_work_model_audit.md` Section C).

Two datasets (BPIC17, BPIC19) are also used by this project's own SuTraN/CRTP-LSTM model roster (A4/A5) — deliberate overlap, since it lets Phase 3's baseline-reproduction check compare against those papers' own reported numbers on the same data. Helpdesk overlaps with ProcessTransformer's (A1) paper for the same reason.

## Splits

All five datasets use the same split methodology (documented per-dataset in `configs/datasets/*.yaml`, `split:` block): cases sorted by start timestamp, then split by position — **not randomly** — into 64% train / 16% validation / 20% test, following the convention used in established PPM benchmarks (e.g. Teinemaa et al. 2019). Sorting by time rather than splitting randomly is required to prevent the temporal/data leakage the spec explicitly warns against (§6): a random split could put a case starting in month 3 into training and a case starting in month 1 into test, letting the model implicitly learn from future process behavior it wouldn't have seen at deployment time.

Every model in the 9-configuration roster (`paper/related_work_model_audit.md`) will consume the same split for a given dataset, generated once by the shared pipeline — no per-model bespoke splitting, per the spec's requirement for identical splits across models.

## Outcome label

Not defined for any dataset — outcome prediction was dropped from this study entirely (see `paper/related_work_model_audit.md` Section C). `outcome_label: null` in every dataset config reflects this; it is not an oversight.

## Done (2026-08-15)

- Downloaded all five raw files into `data/raw/` (each verified against the source's exact reported byte size — no guessed URLs, see `configs/datasets/*.yaml`).
- Implemented the shared preprocessing pipeline: `src/data/schema.py` (canonical case_id/activity/timestamp schema), `src/data/loaders.py` (XES via `pm4py` for BPIC12/17/19/Sepsis, CSV for Helpdesk — both converge to the same schema), `src/data/splits.py` (time-based case-sort split, sha256 split-hash for provenance), `src/data/stats.py` (the full spec §6 descriptive-stats list). Driven by `scripts/prepare_dataset.py --all`, config-driven per `configs/datasets/*.yaml`, writing `data/processed/<dataset>/{train,val,test}.parquet` + `manifest.json` (git commit, software versions, split hash, stats).
- Unit tests (`tests/test_loaders.py`, `tests/test_splits.py`, `tests/test_stats.py`) cover the splitting and stats logic against synthetic data with hand-computed expected values — pass without needing the real (gitignored) datasets.

## Not yet done (tracked in `STATUS.md`)

- Resolving the BPIC19 timestamp data-quality artifact (see note above) with an explicit, documented cleaning rule before Phase 3 training.
- Dataset version/hash recording beyond the split hash already in each manifest — e.g. hashing the raw downloaded file itself for full spec §21 provenance.
