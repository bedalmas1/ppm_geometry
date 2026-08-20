#!/usr/bin/env bash
#
# Phase 6 full-scale sub-phase: train + extract all 9 roster models on the
# 4 datasets not yet covered by Helpdesk (Sepsis, BPIC12, BPIC17, BPIC19),
# one at a time, in dataset-size order.
#
# Bash/Git-Bash equivalent of scripts/run_full_scale_training.ps1 - see that
# file's header for the full rationale (single shared GPU + Phase 3's
# sequential-training policy; this environment's own background processes
# get killed at ~60 minutes regardless of training health, so this is meant
# to be run directly in an interactive terminal, not backgrounded).
#
# Safe to interrupt (Ctrl+C) and re-run: each combination is skipped if its
# checkpoint AND embeddings_test.parquet already exist. MLMME already has
# its own internal epoch-level resume (train_mlmme.py's resume_state.pt) -
# re-running its exact same command picks up training where it left off
# rather than restarting, so an interrupted MLMME run doesn't need special
# handling here either.
#
# Usage:
#   ./scripts/run_full_scale_training.sh --dry-run
#   ./scripts/run_full_scale_training.sh
#   ./scripts/run_full_scale_training.sh --start-index 10

set -u

DRY_RUN=0
START_INDEX=1
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Resolve a working uv executable ('uv' on PATH resolving to a broken
# `python -m uv` shim with no `uv` package installed is a known failure mode
# on this machine - fall back to the known pip --user install location). ---
if command -v uv >/dev/null 2>&1; then
  UV_EXE="uv"
else
  FALLBACK="${APPDATA//\\//}/Python/Python313/Scripts/uv.exe"
  if [ -f "$FALLBACK" ]; then
    UV_EXE="$FALLBACK"
    echo "Note: 'uv' not found on PATH, using $UV_EXE instead."
  else
    echo "ERROR: could not find a working 'uv' executable (checked PATH and $FALLBACK)." >&2
    exit 1
  fi
fi

# --- Model roster: script-name suffix (shared by train_/extract_/results
# dir), config-file prefix, the uv --extra group it needs, and its
# checkpoint filename (used only to detect "already trained" for the skip
# check). Order matches scripts/run_full_scale_training.ps1. ---
MODEL_ORDER=(process_transformer generative_lstm rlhgnn sutran crtp_lstm lupin mlmme controlled_transformer_next controlled_transformer_suffix)

declare -A MODEL_CONFIG_PREFIX=(
  [process_transformer]=pt
  [generative_lstm]=lstm
  [rlhgnn]=rlhgnn
  [sutran]=sutran
  [crtp_lstm]=crtp_lstm
  [lupin]=lupin
  [mlmme]=mlmme
  [controlled_transformer_next]=controlled_transformer_next
  [controlled_transformer_suffix]=controlled_transformer_suffix
)
declare -A MODEL_EXTRA=(
  [process_transformer]=tf
  [generative_lstm]=tf
  [rlhgnn]=torch-dgl
  [sutran]=torch
  [crtp_lstm]=torch
  [lupin]=torch-hf
  [mlmme]=torch
  [controlled_transformer_next]=torch
  [controlled_transformer_suffix]=torch
)
declare -A MODEL_CHECKPOINT=(
  [process_transformer]=checkpoint.weights.h5
  [generative_lstm]=checkpoint.weights.h5
  [rlhgnn]=checkpoint.pt
  [sutran]=checkpoint.pt
  [crtp_lstm]=checkpoint.pt
  [lupin]=checkpoint.pt
  [mlmme]=checkpoint.pt
  [controlled_transformer_next]=checkpoint.pt
  [controlled_transformer_suffix]=checkpoint.pt
)

# Smallest dataset first, per STATUS.md's recommended order.
DATASETS=(sepsis bpic12 bpic17 bpic19)

LOG_DIR="$REPO_ROOT/results/full_scale_training_logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "$1" | tee -a "$LOG_PATH"
}

log "=== Phase 6 full-scale training run started $(date -Iseconds) ==="
log "Log file: $LOG_PATH"

# Piping a Python subprocess's stdout through `tee` makes Python think it
# isn't attached to a terminal, so it switches from line-buffered to
# fully-buffered output (~8KB chunks) - print() calls can then sit unseen
# for a long time (e.g. train_sutran.py's short per-epoch prints can take
# 100+ epochs to fill the buffer). Force unbuffered stdout so progress
# prints show up immediately instead of looking like a hang.
export PYTHONUNBUFFERED=1

TOTAL=$(( ${#DATASETS[@]} * ${#MODEL_ORDER[@]} ))
INDEX=0
SUCCEEDED=()
FAILED=()
SKIPPED=()

for dataset in "${DATASETS[@]}"; do
  for model in "${MODEL_ORDER[@]}"; do
    INDEX=$((INDEX + 1))
    if [ "$INDEX" -lt "$START_INDEX" ]; then
      continue
    fi

    prefix="${MODEL_CONFIG_PREFIX[$model]}"
    extra="${MODEL_EXTRA[$model]}"
    checkpoint_file="${MODEL_CHECKPOINT[$model]}"
    label="[$INDEX/$TOTAL] $model / $dataset"

    config_path="configs/experiments/${prefix}_${dataset}.yaml"
    train_script="experiments/train_${model}.py"
    extract_script="experiments/extract_${model}.py"
    checkpoint_path="results/$dataset/$model/$checkpoint_file"
    embeddings_path="results/$dataset/$model/embeddings_test.parquet"

    if [ "$DRY_RUN" -eq 1 ]; then
      log "$label"
      log "    $UV_EXE run --extra $extra python $train_script $config_path"
      log "    $UV_EXE run --extra $extra python $extract_script $dataset"
      continue
    fi

    if [ -f "$checkpoint_path" ] && [ -f "$embeddings_path" ]; then
      log "$label -- already trained + extracted, skipping."
      SKIPPED+=("$label")
      continue
    fi

    log "$label -- training..."
    "$UV_EXE" run --extra "$extra" python "$train_script" "$config_path" 2>&1 | tee -a "$LOG_PATH"
    train_status=${PIPESTATUS[0]}
    if [ "$train_status" -ne 0 ]; then
      log "$label -- TRAIN FAILED (exit $train_status), skipping extraction, continuing to next combination."
      FAILED+=("$label (train)")
      continue
    fi

    log "$label -- extracting..."
    "$UV_EXE" run --extra "$extra" python "$extract_script" "$dataset" 2>&1 | tee -a "$LOG_PATH"
    extract_status=${PIPESTATUS[0]}
    if [ "$extract_status" -ne 0 ]; then
      log "$label -- EXTRACT FAILED (exit $extract_status), continuing to next combination."
      FAILED+=("$label (extract)")
      continue
    fi

    log "$label -- done."
    SUCCEEDED+=("$label")
  done
done

if [ "$DRY_RUN" -eq 1 ]; then
  log "=== Dry run complete: $TOTAL combinations listed, nothing executed. ==="
  exit 0
fi

log ""
log "=== Run complete $(date -Iseconds) ==="
log "Succeeded: ${#SUCCEEDED[@]}  Skipped (already done): ${#SKIPPED[@]}  Failed: ${#FAILED[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
  log "Failed combinations (see $LOG_PATH for the actual error output):"
  for f in "${FAILED[@]}"; do
    log "  - $f"
  done
fi
