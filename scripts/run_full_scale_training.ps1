<#
.SYNOPSIS
    Phase 6 full-scale sub-phase: train + extract all 9 roster models on the
    4 datasets not yet covered by Helpdesk (Sepsis, BPIC12, BPIC17, BPIC19),
    one at a time, in dataset-size order.

.DESCRIPTION
    Runs the 36 `train_<model>.py` / `extract_<model>.py` command pairs
    listed in STATUS.md's Phase 6 next-steps, sequentially (this project's
    single shared GPU + Phase 3's sequential-training policy - never run two
    of these in parallel). Meant to be run directly in a user terminal, not
    as a background process, since this environment's own background
    processes get killed at ~60 minutes regardless of training health
    (found during A7 MLMME's integration) - an interactive terminal has no
    such limit.

    Safe to interrupt (Ctrl+C) and re-run: each combination is skipped if
    its checkpoint AND embeddings_test.parquet already exist. MLMME already
    has its own internal epoch-level resume (train_mlmme.py's
    resume_state.pt) - re-running its exact same command picks up training
    where it left off rather than restarting, so an interrupted MLMME run
    doesn't need special handling here either.

.PARAMETER DryRun
    Print every command that would run, in order, without executing
    anything. Use this first to sanity-check the plan.

.PARAMETER StartIndex
    1-based index into the full 36-combination plan to start from (after
    the dry run's printed list) - use if you want to skip ahead manually
    instead of relying on the automatic already-done skip check.

.EXAMPLE
    .\scripts\run_full_scale_training.ps1 -DryRun
    .\scripts\run_full_scale_training.ps1
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [int]$StartIndex = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# --- Resolve the uv executable (PATH resolution to a `python -m uv` shim
# with no `uv` package installed is a known failure mode on this machine -
# fall back to the known pip --user install location). ---
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCmd) {
    $UvExe = $uvCmd.Source
} else {
    $fallback = Join-Path $env:APPDATA "Python\Python313\Scripts\uv.exe"
    if (Test-Path $fallback) {
        $UvExe = $fallback
        Write-Host "Note: 'uv' not found on PATH, using $UvExe instead." -ForegroundColor Yellow
    } else {
        throw "Could not find a working 'uv' executable (checked PATH and $fallback). See the PATH-fix options discussed earlier in this session."
    }
}

# --- Model roster: script-name suffix (shared by train_/extract_/results
# dir), the uv --extra group it needs, and its checkpoint filename (used
# only to detect "already trained" for the skip check). ---
$Models = @(
    @{ Name = "process_transformer";            ConfigPrefix = "pt";                           Extra = "tf";        Checkpoint = "checkpoint.weights.h5" }
    @{ Name = "generative_lstm";                 ConfigPrefix = "lstm";                         Extra = "tf";        Checkpoint = "checkpoint.weights.h5" }
    @{ Name = "rlhgnn";                           ConfigPrefix = "rlhgnn";                       Extra = "torch-dgl"; Checkpoint = "checkpoint.pt" }
    @{ Name = "sutran";                           ConfigPrefix = "sutran";                       Extra = "torch";     Checkpoint = "checkpoint.pt" }
    @{ Name = "crtp_lstm";                        ConfigPrefix = "crtp_lstm";                    Extra = "torch";     Checkpoint = "checkpoint.pt" }
    @{ Name = "lupin";                            ConfigPrefix = "lupin";                        Extra = "torch-hf";  Checkpoint = "checkpoint.pt" }
    @{ Name = "mlmme";                            ConfigPrefix = "mlmme";                        Extra = "torch";     Checkpoint = "checkpoint.pt" }
    @{ Name = "controlled_transformer_next";      ConfigPrefix = "controlled_transformer_next";  Extra = "torch";     Checkpoint = "checkpoint.pt" }
    @{ Name = "controlled_transformer_suffix";    ConfigPrefix = "controlled_transformer_suffix";Extra = "torch";     Checkpoint = "checkpoint.pt" }
)

# Smallest dataset first, per STATUS.md's recommended order.
$Datasets = @("sepsis", "bpic12", "bpic17", "bpic19")

# --- Build the full 36-item plan. ---
$Plan = @()
foreach ($dataset in $Datasets) {
    foreach ($model in $Models) {
        $Plan += [PSCustomObject]@{
            Dataset       = $dataset
            Model         = $model.Name
            ConfigPrefix  = $model.ConfigPrefix
            Extra         = $model.Extra
            Checkpoint    = $model.Checkpoint
            TrainScript   = "experiments/train_$($model.Name).py"
            ExtractScript = "experiments/extract_$($model.Name).py"
            ConfigPath    = "configs/experiments/$($model.ConfigPrefix)_$dataset.yaml"
        }
    }
}

$LogDir = Join-Path $RepoRoot "results\full_scale_training_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $LogPath -Value $Message
}

Write-Log "=== Phase 6 full-scale training run started $(Get-Date -Format o) ==="
Write-Log "Log file: $LogPath"

# Piping a Python subprocess's stdout through Tee-Object makes Python think
# it isn't attached to a terminal, so it switches from line-buffered to
# fully-buffered output (~8KB chunks) - print() calls can then sit unseen
# for a long time (e.g. train_sutran.py's short per-epoch prints can take
# 100+ epochs to fill the buffer). Force unbuffered stdout so progress
# prints show up immediately instead of looking like a hang.
$env:PYTHONUNBUFFERED = "1"

$succeeded = @()
$failed = @()
$skipped = @()

for ($i = 0; $i -lt $Plan.Count; $i++) {
    $index = $i + 1
    if ($index -lt $StartIndex) { continue }
    $item = $Plan[$i]
    $label = "[$index/36] $($item.Model) / $($item.Dataset)"

    $checkpointPath = Join-Path $RepoRoot "results\$($item.Dataset)\$($item.Model)\$($item.Checkpoint)"
    $embeddingsPath = Join-Path $RepoRoot "results\$($item.Dataset)\$($item.Model)\embeddings_test.parquet"

    $trainArgs = @("run", "--extra", $item.Extra, "python", $item.TrainScript, $item.ConfigPath)
    $extractArgs = @("run", "--extra", $item.Extra, "python", $item.ExtractScript, $item.Dataset)

    if ($DryRun) {
        Write-Log "$label"
        Write-Log "    $UvExe $($trainArgs -join ' ')"
        Write-Log "    $UvExe $($extractArgs -join ' ')"
        continue
    }

    if ((Test-Path $checkpointPath) -and (Test-Path $embeddingsPath)) {
        Write-Log "$label -- already trained + extracted, skipping."
        $skipped += $label
        continue
    }

    Write-Log "$label -- training..."
    & $UvExe @trainArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Log "$label -- TRAIN FAILED (exit $LASTEXITCODE), skipping extraction, continuing to next combination."
        $failed += "$label (train)"
        continue
    }

    Write-Log "$label -- extracting..."
    & $UvExe @extractArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Log "$label -- EXTRACT FAILED (exit $LASTEXITCODE), continuing to next combination."
        $failed += "$label (extract)"
        continue
    }

    Write-Log "$label -- done."
    $succeeded += $label
}

if ($DryRun) {
    Write-Log "=== Dry run complete: $($Plan.Count) combinations listed, nothing executed. ==="
    return
}

Write-Log ""
Write-Log "=== Run complete $(Get-Date -Format o) ==="
Write-Log "Succeeded: $($succeeded.Count)  Skipped (already done): $($skipped.Count)  Failed: $($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Log "Failed combinations (see $LogPath for the actual error output):"
    foreach ($f in $failed) { Write-Log "  - $f" }
}
