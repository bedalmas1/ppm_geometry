# Cluster migration guide — Phase 6 full-scale training

One-time guide for running the remaining Phase 6 full-scale training (the
same 9-models × 4-datasets work `scripts/run_full_scale_training.sh` drives
locally) on a SLURM-managed compute cluster instead of a laptop. Written
2026-09-13. Companion to [`LAPTOP_MIGRATION.md`](./LAPTOP_MIGRATION.md) (which
covers laptop→laptop moves) — read that file's rationale section first if you
haven't; this file only covers what's *different* about a cluster.

Assumptions baked into this guide (stated so you can tell if they don't hold
for your actual allocation):
- **Scheduler: SLURM** (`sbatch`/`squeue`/`sacct`). Swap the `#SBATCH`
  directives for `#PBS`/`qsub` equivalents if your cluster runs PBS/Torque
  instead — the underlying logic (checkpoint/resume, module loads, `uv sync`)
  doesn't change.
- **Compute nodes have no internet access**, only the login node does. All
  package installs and dataset downloads happen on the login node; compute
  nodes only ever read the already-built `.venv`/`data/` from shared storage.
- **GPU capacity per allocation is unknown** — treated as roughly comparable
  to the old laptop's single 8GB RTX A2000 (one GPU, one model training at a
  time) until you confirm otherwise. See "If you get more than one GPU"
  below for how to parallelize once you know your real allocation.

## What "the same thing" actually is

`scripts/run_full_scale_training.sh` drives 9 models × 4 datasets = 36
(model, dataset) combinations, each one `train_<model>.py` → `extract_<model>.py`,
skipping any combination whose checkpoint + `embeddings_test.parquet` already
exist. **As of 2026-09-13, local progress is:**

| Dataset | Done (9) | Notes |
|---|---|---|
| sepsis | 9/9 | fully done |
| bpic12 | 5/9 | `lupin` has a checkpoint but no extracted embeddings yet (extraction-only, cheap); `mlmme`, `controlled_transformer_next`, `controlled_transformer_suffix` not started |
| bpic17 | 0/9 | not started |
| bpic19 | 0/9 | not started |

So **22 of 36 combinations remain** (21 full train+extract, 1 extract-only).
The cluster's job is to finish exactly these, not to redo Sepsis.

Model → required `uv` extra (from `pyproject.toml`, extras are mutually
exclusive via `[tool.uv] conflicts`):

| Model script suffix | Extra | Checkpoint/resume support |
|---|---|---|
| `process_transformer` | `tf` | **No** (TF, never needed it — fast/small) |
| `generative_lstm` | `tf` | **No** (same) |
| `rlhgnn` | `torch-dgl` | Yes |
| `sutran` | `torch` | Yes |
| `crtp_lstm` | `torch` | Yes |
| `lupin` | `torch-hf` | Yes |
| `mlmme` | `torch` | Yes |
| `controlled_transformer_next` | `torch` | Yes |
| `controlled_transformer_suffix` | `torch` | Yes |

## Part A — What to embed (bring to the cluster)

Copy these from this machine to the cluster's **persistent** project storage
(not a scratch/tmp filesystem subject to automatic purging — check your
cluster's docs for which mount is persistent):

1. **The code, via `git clone`** on the cluster's login node (internet
   access assumed there) — don't `scp` the repo, it's already all on
   `origin/main`:
   ```
   git clone https://github.com/bedalmas1/ppm_geometry.git
   cd ppm_geometry
   ```
2. **`data/processed/` (~42 MB)** — the prepared parquet splits + `manifest.json`
   files with the split-hash. **Copy this, don't regenerate it on the
   cluster** — regenerating guarantees the same *code path* but not
   byte-identical output unless `pm4py`/pandas versions match exactly, and
   spec §21's provenance requirement wants one recorded split, not a second
   independently-derived one. `scp -r data/processed <user>@<cluster>:.../ppm_geometry/`
   (or `rsync -av`).
3. **`data/raw/` (~730 MB) — optional.** Since the login node has internet,
   it's just as easy to re-download from the 4TU links in
   `configs/datasets/*.yaml` on the cluster directly (see Part B step 2) as
   to transfer it. Only copy it across if bandwidth/time favors that.
4. **`results/` (~1.8 GB currently)** — every already-trained checkpoint
   (`checkpoint.pt` / `.weights.h5`), each model's `resume_state.pt`/`.json`
   (needed for MLMME-style mid-training resume to actually resume rather
   than restart), extracted `embeddings_test.parquet` files, and
   `results/full_scale_training_logs/`. **This is the irreplaceable one** —
   it's the actual training progress, including the Sepsis-complete and
   BPIC12-partial state in the table above. Preserve the relative paths
   (`results/bpic12/process_transformer/...`) exactly.

Do **not** copy `.venv/` — rebuild it on the cluster (Part B step 3). It's
pinned to specific torch/DGL/TensorFlow builds tied to this machine's CPU
(no GPU) and won't be right for the cluster's actual GPU/CUDA anyway.

## Part B — Set up on the cluster

Run all of this **on the login node** (needs internet; compute nodes don't
get it).

1. **Confirm GPU/CUDA on a compute node** before installing anything GPU-specific:
   ```
   srun --partition=<gpu-partition> --gres=gpu:1 --pty nvidia-smi
   ```
   Note the GPU model, VRAM, and CUDA version the driver reports. This
   determines whether `pyproject.toml`'s current CPU-only `torch = ["torch>=2.3"]`
   needs to be redirected to a CUDA wheel index again, the way it briefly was
   for the old RTX A2000 laptop (see `STATUS.md`'s 2026-08-19 decision log
   entry: `[[tool.uv.index]]` pointing the `torch` extra at
   `download.pytorch.org/whl/cu124`, then reverted 2026-09-13 when that
   laptop was retired). **`torch-dgl` (RLHGNN) needs extra care**: it's
   hard-pinned to `torch==2.3.0`/`dgl==2.2.1` because that's the newest
   combination DGL ships a precompiled `graphbolt` binary for — confirm that
   pin actually has a CUDA build compatible with the cluster's driver before
   assuming it will "just work" (this was flagged as unverified even on the
   old laptop, see `STATUS.md` Open questions).
2. **Get the data.** Either restore the copied `data/raw/`+`data/processed/`
   at the same relative paths, or (since the login node has internet) run
   `uv run python scripts/prepare_dataset.py --all` fresh — but if you
   regenerate rather than copy `data/processed/`, diff the resulting
   `manifest.json` split-hashes against the copies from Part A to confirm
   they match before trusting any training run against them.
3. **Build one `.venv` per framework extra, not one shared venv reused
   across extras.** This matters more here than it did locally: locally,
   training was strictly sequential (one model at a time), so `uv run --extra X`
   re-syncing the single `.venv` before each run never raced with anything.
   On a cluster, if you ever run two SLURM jobs concurrently that need
   different extras (e.g. a `tf` job and a `torch-dgl` job at the same time),
   two concurrent `uv sync`/`uv run --extra` calls rewriting the *same*
   `.venv` directory will corrupt it. Give each extra its own environment via
   `UV_PROJECT_ENVIRONMENT`:
   ```
   UV_PROJECT_ENVIRONMENT=.venv-tf        uv sync --extra tf
   UV_PROJECT_ENVIRONMENT=.venv-torch     uv sync --extra torch
   UV_PROJECT_ENVIRONMENT=.venv-torch-hf  uv sync --extra torch-hf
   UV_PROJECT_ENVIRONMENT=.venv-torch-dgl uv sync --extra torch-dgl
   ```
   (If you're certain you'll only ever run one job at a time on this
   allocation, a single shared `.venv` re-synced per `uv run --extra X` call —
   exactly what `run_full_scale_training.sh` already does — is fine and
   simpler. Only split environments once you parallelize.)
4. **Smoke-test each environment actually imports and sees the GPU**, don't
   assume `uv sync` succeeding means it works (this bit twice locally: TF/CPython
   3.13 wheels, and DGL's `graphbolt` extension needing an exact torch build —
   see `STATUS.md` Phase 0/2). From a GPU-allocated interactive session:
   ```
   UV_PROJECT_ENVIRONMENT=.venv-torch uv run --extra torch python -c \
     "import torch; print(torch.__version__, torch.cuda.is_available())"
   UV_PROJECT_ENVIRONMENT=.venv-torch-dgl uv run --extra torch-dgl python -c \
     "import torch, dgl; print(torch.__version__, dgl.__version__)"
   UV_PROJECT_ENVIRONMENT=.venv-tf uv run --extra tf python -c \
     "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   UV_PROJECT_ENVIRONMENT=.venv-torch-hf uv run --extra torch-hf python -c \
     "import torch, transformers; print(transformers.__version__)"
   ```
5. **Sanity-check resume actually resumes**, on one already-in-progress
   model, before trusting the batch job: re-run e.g. BPIC12's `lupin`
   extraction (`experiments/extract_lupin.py bpic12`) or a torch model's
   train script against its existing config and confirm training scripts
   pick up from a non-zero epoch, not epoch 0.

## Part C — The SLURM job

Because compute-node walltimes are finite (commonly 24–48h) and the two TF
models (`process_transformer`, `generative_lstm`) have **no** checkpoint/resume,
structure the job so a walltime cutoff never wastes more than one
TF-sized run (these are the fast/small models — "never approached an hour"
even on Helpdesk per `STATUS.md`), and every torch/DGL model resumes cleanly
via its own `resume_state.pt`/`.json` no matter when it's interrupted.

**Default: keep it sequential**, just moved onto the cluster's GPU node
instead of the laptop's, driven by the existing
`scripts/run_full_scale_training.sh` unchanged (it's already portable POSIX
shell) with `--start-index` used to resume across job resubmissions. A
self-resubmitting wrapper handles the walltime limit:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=ppm-geometry-fullscale
#SBATCH --partition=<gpu-partition>       # fill in
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8                 # fill in per cluster policy
#SBATCH --mem=32G                         # fill in
#SBATCH --time=24:00:00                   # fill in to your cluster's max
#SBATCH --output=results/full_scale_training_logs/slurm_%j.out

set -eu
cd "$SLURM_SUBMIT_DIR"

module load cuda/<version-matching-Part-B-step-1>   # fill in
# module load cudnn/<version> if your cluster separates it

# Record provenance (spec §21) alongside the run's own log output.
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_JOB_NODELIST GIT_SHA=$(git rev-parse HEAD)"

# Each combination's `uv run --extra <extra>` call resolves its own
# environment; if you split per-extra venvs (Part B step 3), export
# UV_PROJECT_ENVIRONMENT per model instead — run_full_scale_training.sh
# doesn't currently vary this per model, so either keep one shared venv
# (fine while this job stays sequential) or fork the script if you split.
bash scripts/run_full_scale_training.sh

REMAINING=$(bash scripts/run_full_scale_training.sh --dry-run | grep -c '\[.*\]' || true)
if [ "$REMAINING" -gt 0 ] && [ "${SLURM_RESTART_COUNT:-0}" -lt 20 ]; then
  sbatch --export=ALL "$0"
fi
```

Notes on this template:
- `run_full_scale_training.sh` already skips any combination with both a
  checkpoint and `embeddings_test.parquet` present, so re-running it after a
  walltime cutoff naturally continues rather than restarting finished work —
  the self-resubmit above just automates "run it again" instead of you
  watching `squeue`.
- The `--dry-run` grep-count check for "is anything left" is a placeholder —
  `run_full_scale_training.sh --dry-run` doesn't currently print a clean
  "N remaining" number, so this either needs a small script change (have
  `--dry-run` print a final count) or you resubmit unconditionally and let
  the very last resubmission finish in well under its time budget and exit
  0 with nothing to do.
- The `SLURM_RESTART_COUNT` guard is just a safety cap against infinite
  resubmission if something is silently and permanently failing — check
  `results/full_scale_training_logs/` for repeated failures on the same
  combination before assuming it'll resolve itself.
- `PYTHONUNBUFFERED=1` is already set inside `run_full_scale_training.sh` —
  no need to set it again here (this fixed a real "looks hung, isn't" bug
  locally with SuTraN's output, per `STATUS.md`'s 2026-08-17 entry).

## If you get more than one GPU

Once you actually know your allocation supports concurrent GPU use (multiple
GPUs per node, or cheap to submit many single-GPU jobs), the sequential
constraint from `STATUS.md`'s Phase 3 decision (*"single shared 8GB GPU
would make parallel runs contend for VRAM"*) no longer necessarily applies —
that was a laptop-specific constraint, not a scientific one. The 36
combinations are embarrassingly parallel across (model, dataset) pairs. To
parallelize:

1. Split per-extra `.venv`s as in Part B step 3 (mandatory once jobs run
   concurrently, to avoid two jobs re-syncing one shared `.venv`).
2. Turn the 22 remaining combinations into a SLURM array — e.g. a small
   Python/bash generator emitting one `(model, dataset)` pair per array task
   index, each task running just that pair's
   `uv run --extra <extra> python experiments/train_<model>.py <config>`
   then `extract_<model>.py <dataset>`, with `--gres=gpu:1` per task.
3. Still respect one caveat: the original Phase 3 compute-timing
   measurements (e.g. A7 MLMME's ~135s/epoch on Helpdesk, ~450s/epoch on
   Sepsis pre-fix — see `STATUS.md`) were taken on the old single-GPU
   laptop. If you use those numbers to size SLURM `--time` requests on
   different GPU hardware, treat them as rough starting points, not
   guarantees — re-measure once the first cluster run of each model
   completes and adjust.
4. `results/full_scale_training_logs/` is currently one file per
   sequential run; if you parallelize, either give each array task its own
   log filename (avoid concurrent writers to one file) or rely on each
   SLURM task's own `--output=%A_%a.out`.

## Part D — After training

1. **Copy `results/` back off the cluster** to persistent storage (this
   laptop, external drive, or wherever else) — don't assume cluster project
   storage is retained indefinitely; check your cluster's purge policy.
2. **Update `STATUS.md`** per `CLAUDE.md`'s session-end convention: tick off
   the full-scale sub-phase once all 36 combinations are done, log the
   cluster migration as a dated decision-log entry (including the actual GPU
   model/CUDA version used, since that's part of spec §21's provenance
   requirement), and update "Next steps" to point at Phase 6's remaining
   full-scale geometry analysis (`experiments/run_geometry_pilot.py` →
   full-scale runner, per `STATUS.md`'s existing next-step #2).
