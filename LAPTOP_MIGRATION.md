# Laptop migration checklist

One-time guide for moving this project to a new laptop without recomputing any
training. Written 2026-08-23, when Phase 6's full-scale training (36 runs
across 9 models x 4 remaining datasets) is in progress, run manually by the
user in a separate terminal per `STATUS.md`'s execution-strategy note.

Why this is safe: `git status` shows the working tree matches `origin/main`
exactly — all code, configs, and the checkpoint/resume logic (added in the
"Phase 6 full-scale prep" commit) are already pushed. The only things at risk
are gitignored, locally-generated artifacts: prepared data and training
checkpoints/resume state. Nothing is lost mid-training as long as it's copied
across, because checkpoint/resume already externalizes full training state
(weights, optimizer, RNG, epoch counter) to disk on every epoch.

## Part A — on this (old) laptop, before giving it up

1. **Don't kill training mid-write.** Let the current run reach its next
   epoch checkpoint, or just Ctrl-C between epochs — resume is designed for
   exactly this. Don't `rm`/reset anything.
2. **Confirm git is clean and pushed:**
   ```
   git status
   git push
   ```
   Should show `nothing to commit, working tree clean` and no local commits
   ahead of `origin/main`. If anything's uncommitted, decide with a fresh
   session whether it belongs in git before moving on — don't force-push or
   discard.
3. **Copy these paths to external media (USB / external drive / cloud
   drive / LAN share) — measured sizes as of 2026-08-23, ~1.8 GB total:**
   - `results/` (~1.0 GB) — every model's `checkpoint.pt` / `.weights.h5` /
     `resume_model.keras`, plus the `resume_state.pt` / `resume_state.json`
     files the checkpoint/resume feature writes, plus extracted embeddings,
     manifests, and `results/full_scale_training_logs/`. **This is the
     irreplaceable one** — it's the actual training progress.
   - `data/processed/` (~42 MB) — prepared parquet splits + manifests with
     the split-hash. Copy rather than regenerate, to guarantee the hash
     matches exactly.
   - `data/raw/` (~730 MB, optional) — the original downloaded XES/CSV logs.
     Not required (re-downloadable from the 4TU links in
     `configs/datasets/*.yaml`), but saves a re-download.
   - `experiments/script` (untracked, not in git) — your manual progress
     ledger for the 36-run batch: lines prefixed `OK` are already done.
     **Don't lose this**, it's the only record of which runs are finished.
   Preserve the relative paths (e.g. `results/bpic12/process_transformer/...`)
   so nothing needs re-pointing on the other end.
4. Do **not** bother copying `.venv/` — rebuild it fresh on the new laptop
   (see Part B, step 3). It's pinned to specific torch/DGL/TensorFlow builds
   that are GPU/CUDA-build-specific (see `STATUS.md` Phase 2's dgl pinning
   saga) and may not be valid on different hardware anyway.

## Part B — on the new laptop, after setup

1. **Clone the repo:**
   ```
   git clone https://github.com/bedalmas1/ppm_geometry.git
   ```
2. **Restore the copied artifacts** into the clone at the same relative
   paths: `results/`, `data/processed/`, `data/raw/` (if copied),
   `experiments/script`.
3. **Rebuild the environment from `uv.lock`** (don't reuse the old
   `.venv/`): install `uv`, then per framework group as needed —
   ```
   uv sync --extra tf
   uv sync --extra torch
   uv sync --extra torch-hf
   uv sync --extra torch-dgl
   ```
   Check `nvidia-smi` on the new machine first. If the GPU differs from the
   old RTX A2000 8GB (or there's no GPU), the Phase 3 compute-timing numbers
   and any batch sizes tuned to 8 GB VRAM may no longer hold, and A3
   RLHGNN's pinned `torch==2.3.0` (needed for the DGL `graphbolt` wheel) is
   tied to a specific CUDA build — verify it still installs and imports
   before assuming training will "just work." Re-run the dependency-group
   smoke test described in `STATUS.md` Phase 0/2 if anything looks off.
4. **Fix `experiments/script`'s interpreter path.** It currently hardcodes
   this laptop's `uv.exe` location
   (`C:\Users\Benjamin.Dalmas\AppData\Roaming\Python\Python313\Scripts\uv.exe`).
   On the new laptop, either update every line to the new `uv.exe` path, or
   simplify each line to just `uv run --extra ... python ...` if `uv` is on
   `PATH` — keep the `OK` markers as-is, they're your progress record.
5. **Sanity-check resume actually resumes**, on one in-progress model, before
   trusting anything: re-run its `experiments/train_*.py` command with its
   existing config and confirm the log shows it resuming from a non-zero
   epoch (reading `resume_state.pt`/`.json`), not restarting from epoch 0.
   Also spot-check `data/processed/*/manifest.json`'s split hash matches
   what a fresh `scripts/prepare_dataset.py` run would produce, if you want
   extra confidence the copied data is intact.
6. **Only once verified working**, treat the old laptop's local copies as
   disposable.
7. **Update `STATUS.md`**: note the migration date/laptop change in the
   decision log, and update "Next steps" if the new hardware changes
   anything about how the remaining ~36 runs should proceed.

## Quick reference — what NOT to worry about

- Code, configs, `PLAN.md`, `STATUS.md`, `uv.lock` — already in git, already
  pushed, arrives automatically via `git clone`.
- `.venv/` — intentionally rebuilt, not copied.
- Anything under `__pycache__/` — regenerated automatically, ignore.
