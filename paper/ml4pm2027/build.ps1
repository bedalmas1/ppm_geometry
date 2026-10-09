# Windows build helper: regenerate tables from results/ and data/, then compile.
# Usage:  .\build.ps1              draft (TODO boxes shown)           -> build\main.pdf
#         .\build.ps1 -Final       submission version (no TODO boxes) -> build\submission.pdf
#         .\build.ps1 -SkipAssets  compile only
param([switch]$SkipAssets, [switch]$Final)
$ErrorActionPreference = "Stop"
$paperDir = $PSScriptRoot
$repoRoot = Resolve-Path (Join-Path $paperDir "..\..")
if (-not $SkipAssets) {
    Push-Location $repoRoot
    try { & "$repoRoot\.venv\Scripts\python.exe" "paper\ml4pm2027\scripts\make_assets.py" } finally { Pop-Location }
}
$job = if ($Final) { "submission" } else { "main" }
Push-Location $paperDir
try { latexmk "$job.tex" } finally { Pop-Location }
Write-Host "PDF: $paperDir\build\$job.pdf  (ML4PM 2027 limit: 12 pages incl. figures, references and appendices)"
