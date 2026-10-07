#!/usr/bin/env python3
"""Phase 6 full-scale training on a large multi-core / single-big-GPU server.

Parallel counterpart of scripts/run_full_scale_training.sh. That script is
strictly sequential because the old laptop had one 8GB GPU; on a server with
many cores and a large GPU, the 36 (model, dataset) combinations are
embarrassingly parallel, so this runs several at once in two pools:

  - GPU pool: sutran, crtp_lstm, lupin, mlmme, controlled_transformer_{next,suffix}.
    Several small models share the one GPU concurrently.
  - CPU pool: process_transformer, generative_lstm (TF) and rlhgnn (DGL).
    These run with CUDA_VISIBLE_DEVICES="" on purpose: the `torch-dgl`
    extra's PyPI dgl==2.2.1 wheel is CPU-only while its pinned torch==2.3.0
    is a CUDA build on Linux, so letting it see the GPU makes
    `dgl.batch(...).to("cuda")` crash; TF would otherwise grab the whole GPU's
    memory up front and starve the torch jobs sharing it.

Each combination is train_<model>.py then extract_<model>.py, exactly as in
the sequential script (same configs, same batch sizes; nothing about the
training itself changes, only how many run at once).

"Already done" is decided by EITHER:
  - configs/full_scale_completed.txt (committed to git): combinations whose
    results live on another machine and must not be retrained here; or
  - a local results/<dataset>/<model>/ with both checkpoint and
    embeddings_test.parquet (i.e. finished earlier on this machine).
Regenerate the committed list from a machine that has results/ with
`--write-completed`.

Each uv extra gets its own venv (.venv-<extra>, via UV_PROJECT_ENVIRONMENT):
the extras are mutually exclusive, and concurrent `uv run --extra X` calls
re-syncing one shared .venv would corrupt it.

Stdlib only, so run it with any python3 (no venv needed):
  python3 scripts/run_full_scale_server.py --dry-run
  python3 scripts/run_full_scale_server.py --setup-only
  python3 scripts/run_full_scale_server.py --gpu-jobs 6 --cpu-jobs 3
  python3 scripts/run_full_scale_server.py --write-completed   # on the laptop
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPLETED_PATH = REPO_ROOT / "configs" / "full_scale_completed.txt"

# model -> (config prefix, uv extra, checkpoint filename, pool)
# Order within each pool is roughly slowest-first (MLMME is the known
# bottleneck, see STATUS.md), so the longest jobs start earliest and the
# overall run isn't left waiting on one late-started straggler.
MODELS = {
    "mlmme": ("mlmme", "torch", "checkpoint.pt", "gpu"),
    "lupin": ("lupin", "torch-hf", "checkpoint.pt", "gpu"),
    "sutran": ("sutran", "torch", "checkpoint.pt", "gpu"),
    "controlled_transformer_suffix": ("controlled_transformer_suffix", "torch", "checkpoint.pt", "gpu"),
    "crtp_lstm": ("crtp_lstm", "torch", "checkpoint.pt", "gpu"),
    "controlled_transformer_next": ("controlled_transformer_next", "torch", "checkpoint.pt", "gpu"),
    "rlhgnn": ("rlhgnn", "torch-dgl", "checkpoint.pt", "cpu"),
    "generative_lstm": ("lstm", "tf", "checkpoint.weights.h5", "cpu"),
    "process_transformer": ("pt", "tf", "checkpoint.weights.h5", "cpu"),
}
# Largest dataset first for the same longest-job-first reason.
DATASETS = ["bpic19", "bpic17", "bpic12", "sepsis"]

_print_lock = threading.Lock()


def say(msg: str, log_file: Path | None = None) -> None:
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    with _print_lock:
        print(line, flush=True)
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def is_done_locally(dataset: str, model: str) -> bool:
    run_dir = REPO_ROOT / "results" / dataset / model
    return (run_dir / MODELS[model][2]).is_file() and (run_dir / "embeddings_test.parquet").is_file()


def read_completed() -> set[tuple[str, str]]:
    if not COMPLETED_PATH.is_file():
        return set()
    done = set()
    for raw in COMPLETED_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            dataset, model = line.split()
            done.add((dataset, model))
    return done


def write_completed() -> None:
    done = [(d, m) for d in DATASETS for m in MODELS if is_done_locally(d, m)]
    header = (
        "# (dataset, model) combinations already trained + extracted on another machine.\n"
        "# scripts/run_full_scale_server.py skips these even when results/ is absent.\n"
        "# Regenerate with: python3 scripts/run_full_scale_server.py --write-completed\n"
        f"# Generated {dt.date.today().isoformat()} on {socket.gethostname()}.\n"
    )
    COMPLETED_PATH.write_text(header + "".join(f"{d} {m}\n" for d, m in done), encoding="utf-8")
    print(f"Wrote {len(done)} completed combinations to {COMPLETED_PATH.relative_to(REPO_ROOT)}")


def venv_env(extra: str, threads: int, cpu_only: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["UV_PROJECT_ENVIRONMENT"] = str(REPO_ROOT / f".venv-{extra}")
    env["PYTHONUNBUFFERED"] = "1"
    # Without these, every process sizes its thread pools to all cores, and N
    # concurrent jobs x 128 threads each thrash instead of speeding up.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        env[var] = str(threads)
    env["TF_NUM_INTEROP_THREADS"] = "2"
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def setup_envs(uv: str, extras: set[str], log_dir: Path, main_log: Path) -> None:
    """Build each needed per-extra venv from the lockfile, record its exact
    package set for provenance (spec §21)."""
    for extra in sorted(extras):
        say(f"Syncing .venv-{extra} ...", main_log)
        env = venv_env(extra, threads=1, cpu_only=False)
        rc = run_logged([uv, "sync", "--frozen", "--extra", extra], env, log_dir / f"setup_{extra}.log")
        if rc != 0:
            sys.exit(f"uv sync for extra '{extra}' failed, see {log_dir / f'setup_{extra}.log'}")
        with (log_dir / f"packages_{extra}.txt").open("w", encoding="utf-8") as f:
            subprocess.run([uv, "pip", "freeze"], cwd=REPO_ROOT, env=env, stdout=f, check=False)


def check_gpu(uv: str, extras: set[str], log_dir: Path, main_log: Path) -> bool:
    """True iff every GPU-pool extra's torch actually sees CUDA. On Linux
    uv.lock installs torch 2.13.0+cu126 (needs NVIDIA driver >= 525); with
    an older driver, or no GPU visible, torch silently falls back to CPU."""
    ok = True
    probe = "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
    for extra in sorted(extras):
        out = subprocess.run(
            [uv, "run", "--no-sync", "python", "-c", probe],
            cwd=REPO_ROOT, env=venv_env(extra, threads=1, cpu_only=False),
            capture_output=True, text=True,
        )
        say(f"GPU check .venv-{extra}: {out.stdout.strip() or out.stderr.strip()[-300:]}", main_log)
        ok &= out.returncode == 0 and " True " in f" {out.stdout.strip()} "
    return ok


def run_one(uv: str, dataset: str, model: str, threads: int, log_dir: Path, main_log: Path) -> tuple[str, str, str, float]:
    prefix, extra, _, pool = MODELS[model]
    env = venv_env(extra, threads, cpu_only=(pool == "cpu"))
    log_path = log_dir / f"{dataset}_{model}.log"
    start = time.monotonic()

    say(f"START  {dataset}/{model} ({pool}, {threads} threads) -> {log_path.name}", main_log)
    rc = run_logged([uv, "run", "--no-sync", "python", f"experiments/train_{model}.py",
                     f"configs/experiments/{prefix}_{dataset}.yaml"], env, log_path)
    if rc != 0:
        status = f"TRAIN_FAILED({rc})"
    else:
        rc = run_logged([uv, "run", "--no-sync", "python", f"experiments/extract_{model}.py", dataset], env, log_path)
        status = "ok" if rc == 0 else f"EXTRACT_FAILED({rc})"

    elapsed = time.monotonic() - start
    say(f"{'DONE  ' if status == 'ok' else 'FAILED'} {dataset}/{model}: {status} after {elapsed / 3600:.2f}h", main_log)
    return dataset, model, status, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="list what would run and exit")
    parser.add_argument("--setup-only", action="store_true", help="build venvs + GPU check, then exit")
    parser.add_argument("--write-completed", action="store_true", help=f"rewrite {COMPLETED_PATH.name} from local results/ and exit")
    parser.add_argument("--gpu-jobs", type=int, default=6, help="concurrent jobs sharing the GPU (default 6)")
    parser.add_argument("--cpu-jobs", type=int, default=3, help="concurrent CPU-only jobs (default 3)")
    parser.add_argument("--gpu-threads", type=int, default=8, help="CPU threads per GPU job (default 8)")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU threads per CPU job (default: remaining cores / cpu-jobs)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="run GPU-pool models even if torch can't see CUDA")
    args = parser.parse_args()

    if args.write_completed:
        write_completed()
        return

    completed = read_completed()
    tasks, skipped = [], []
    for dataset in [d for d in DATASETS if d in args.datasets]:
        for model in [m for m in MODELS if m in args.models]:
            if (dataset, model) in completed:
                skipped.append(f"{dataset}/{model} (done elsewhere, {COMPLETED_PATH.name})")
            elif is_done_locally(dataset, model):
                skipped.append(f"{dataset}/{model} (done here, results/)")
            else:
                tasks.append((dataset, model))
    gpu_tasks = [t for t in tasks if MODELS[t[1]][3] == "gpu"]
    cpu_tasks = [t for t in tasks if MODELS[t[1]][3] == "cpu"]

    n_cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    cpu_threads = args.cpu_threads or max(1, (n_cores - args.gpu_jobs * args.gpu_threads) // max(1, args.cpu_jobs))

    print(f"Skipping {len(skipped)}:")
    for s in skipped:
        print(f"  - {s}")
    print(f"To run: {len(gpu_tasks)} GPU-pool x{args.gpu_jobs} ({args.gpu_threads} thr each), "
          f"{len(cpu_tasks)} CPU-pool x{args.cpu_jobs} ({cpu_threads} thr each), {n_cores} cores visible")
    for dataset, model in gpu_tasks + cpu_tasks:
        print(f"  - [{MODELS[model][3]}] {dataset}/{model}  (extra: {MODELS[model][1]})")
    if args.dry_run or (not tasks and not args.setup_only):
        return

    uv = shutil.which("uv")
    if uv is None:
        sys.exit("ERROR: 'uv' not on PATH (install: curl -LsSf https://astral.sh/uv/install.sh | sh)")

    log_dir = REPO_ROOT / "results" / "full_scale_training_logs" / f"server_{dt.datetime.now():%Y%m%d_%H%M%S}"
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log = log_dir / "run.log"

    # Provenance for the whole run (spec §21): commit, host, GPU/driver.
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    git_dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    say(f"host={socket.gethostname()} cores={n_cores} git={git_sha}{' (DIRTY)' if git_dirty else ''}", main_log)
    if shutil.which("nvidia-smi"):
        with (log_dir / "nvidia_smi.txt").open("w", encoding="utf-8") as f:
            subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT, check=False)

    extras = {MODELS[m][1] for _, m in tasks} if tasks else {v[1] for v in MODELS.values()}
    setup_envs(uv, extras, log_dir, main_log)
    gpu_extras = {MODELS[m][1] for _, m in gpu_tasks} if tasks else {"torch", "torch-hf"}
    if gpu_extras and not check_gpu(uv, gpu_extras, log_dir, main_log) and not args.allow_cpu_fallback:
        sys.exit("ERROR: torch cannot see the GPU (see GPU check lines above). Check nvidia-smi works in "
                 "this shell, CUDA_VISIBLE_DEVICES isn't set to empty, and the driver is >= 525 (the locked "
                 "Linux torch is a CUDA 12.6 build). Or pass --allow-cpu-fallback to train on CPU anyway.")
    if args.setup_only:
        return

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.gpu_jobs)) as gpu_pool, \
         ThreadPoolExecutor(max_workers=max(1, args.cpu_jobs)) as cpu_pool:
        futures = [gpu_pool.submit(run_one, uv, d, m, args.gpu_threads, log_dir, main_log) for d, m in gpu_tasks]
        futures += [cpu_pool.submit(run_one, uv, d, m, cpu_threads, log_dir, main_log) for d, m in cpu_tasks]
        for fut in futures:
            results.append(fut.result())

    with (log_dir / "summary.tsv").open("w", encoding="utf-8") as f:
        f.write("dataset\tmodel\tstatus\thours\n")
        for dataset, model, status, elapsed in results:
            f.write(f"{dataset}\t{model}\t{status}\t{elapsed / 3600:.2f}\n")
    failed = [r for r in results if r[2] != "ok"]
    say(f"=== Finished: {len(results) - len(failed)} ok, {len(failed)} failed. Summary: {log_dir / 'summary.tsv'}", main_log)
    for dataset, model, status, _ in failed:
        say(f"  FAILED {dataset}/{model}: {status} (see {dataset}_{model}.log)", main_log)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
